function [t_hit, log] = spinn_mechanical_armNN_v2(params25, model_path, opts)
% NN v2 inference core (robust against "downward drop")
% 控制律：几何最速方向 + 抗重力混合 + fill-power + 先总功率后分轴硬帽 + 功率→力矩 + 动力学推进
% 兜底增强：
%   - dir_gravity_mix：把 +G(q) 混入方向，起步更抗下坠
%   - sign_guard：若最终扭矩与 J^T r_hat 夹角>90°，翻转扭矩方向
%   - bootstrap：前 N 步覆盖 NN 权重，用 ratio/equal，并强制肩轴最小占比
%
% 依赖：computeDynamics.m；HNN 仅用于构造动态特征（不直接控臂）

    % --------- 0) 参数与默认 ---------
    assert(isvector(params25) && numel(params25)==25, 'params25 必须是 1x25 向量');
    p25 = params25(:).';
    if nargin < 3 || isempty(opts), opts = struct(); end

    def = struct();
    % 仿真
    def.dt = 0.002; def.t_final = 6.0; def.radius = 0.03;
    def.g = 9.81; def.L = [0.24 0.214 0.324];
    def.useGPU = []; def.verbose = true;
    % 控制律（与 v1 对齐的核心）
    def.k_dir_max   = 25;      % 扭矩种子强度（关键）
    def.omega_eps   = 1e-3;    % 防除零
    def.fill_power  = true;
    def.fill_alpha  = 0.90;
    def.p_boost_max = 10.0;    % 起步弱就放大一点
    def.w_floor     = 0.12;    % 提高下界，防某轴塌权
    def.Pmax_gate   = [];
    % 兜底增强
    def.dir_gravity_mix = 0;      % [0~1] 抗重力混合系数（建议 0.2~0.5）
    def.sign_guard      = true;      % 方向一致性保护
    def.bootstrap_steps = 180;       % 前 N 步覆盖 NN 权重
    def.bootstrap_mode  = 'ratio';   % 'ratio' | 'equal'
    def.shoulder_min    = 0.18;      % 肩轴最小占比（≥ w_floor）
    % HNN 模型路径（来自 v2 训练产物 meta 或显式传入）
    def.hnn_model_path  = '';
    opts = merge_opts(def, opts);

    if isempty(opts.useGPU)
        if exist('canUseGPU','file')==2, opts.useGPU=canUseGPU(); else, opts.useGPU=false; end
    end
    if ~isfile(model_path)
        error('找不到 v2 学生模型：%s', model_path);
    end

    % --------- 1) 载入 v2 学生模型（NN 预测 w） ---------
    M = load(model_path);
    need = {'trainedNet','muX','sigmaX','muY','sigmaY'};
    for i=1:numel(need), if ~isfield(M,need{i}), error('v2 模型缺字段：%s', need{i}); end, end
    net  = M.trainedNet;
    muX  = M.muX(:).';   sX = M.sigmaX(:).';
    muY  = M.muY(:).';   sY = M.sigmaY(:).';

    % --------- 2) 载入 HNN（构造 F10 动态特征） ---------
    hnn_path = '';
    if isfield(M,'meta') && isfield(M.meta,'hnn') && isfield(M.meta.hnn,'model_path') ...
            && ~isempty(M.meta.hnn.model_path)
        hnn_path = M.meta.hnn.model_path;
    elseif ~isempty(opts.hnn_model_path)
        hnn_path = opts.hnn_model_path;
    end
    assert(~isempty(hnn_path) && isfile(hnn_path), 'HNN 模型路径缺失（meta.hnn.model_path 或 opts.hnn_model_path）');
    H = load(hnn_path);
    if isfield(H,'model') && isstruct(H.model) && isfield(H.model,'dlnet')
        dlnH = H.model.dlnet; muH = H.model.muX; sH = H.model.sigmaX;
    else
        dlnH = H.dlnet;       muH = H.muX;       sH = H.sigmaX;
    end
    muHv = dlarray(single(muH(:)),'CB'); sHv = dlarray(single(sH(:)),'CB');
    if opts.useGPU, muHv=gpuArray(muHv); sHv=gpuArray(sHv); end

    % --------- 3) 解包 25 维输入（与 v1 一致） ---------
    m        = p25([1 2 3]);
    dq       = p25([4 5 6]).';
    dampNom  = p25([7 9 11]);
    zetaNom  = p25([8 10 12]);
    q0d      = p25([13 14 15]);
    qTd      = p25([16 17 18]);
    Pmax     = p25(22);
    Prated   = p25([23 24 25]);

    q  = deg2rad(q0d(:));
    qT = deg2rad(qTd(:));
    L  = opts.L;  g = opts.g;

    [xT,yT] = fk_end(qT, L);

    % --------- 4) 日志 ---------
    tvec = (0:opts.dt:opts.t_final).';
    nT = numel(tvec);
    log = struct('t',tvec,'q',nan(nT,3),'dq',nan(nT,3),'w',nan(nT,3), ...
                 'pow_des',zeros(nT,3),'pow_lim',zeros(nT,3),'p',zeros(nT,3), ...
                 'Pmax',zeros(nT,1),'Pmax_eff',zeros(nT,1),'reached',false(nT,1), ...
                 'dEE',nan(nT,1),'v',zeros(nT,1),'hit',false);
    log.q(1,:)=q.'; log.dq(1,:)=dq.';
    [x0,y0] = fk_end(q,L); log.dEE(1)=hypot(x0-xT,y0-yT);
    if log.dEE(1) <= opts.radius
        t_hit=0; log.reached(1)=true; log.hit=true; return;
    end

    % OU 阻尼状态
    ou_xi = zeros(3,1);

    % --------- 5) 主循环 ---------
    for k = 2:nT
        % 5.1 末端方向与几何方向
        [xE,yE]=fk_end(q,L);
        r = [xT-xE; yT-yE]; nr=norm(r); if nr>0, r=r/nr; else, r=[0;0]; end
        J = jacobian_planar(q,L);
        dir_geo = J.' * r;         % 纯几何方向 J^T r_hat

        % 5.2 抗重力混合（稳住起步）
        [Mq,Cq,Gq] = computeDynamics(m(1),m(2),m(3), L(1),L(2),L(3), g, q, dq);
        if opts.dir_gravity_mix>0
            g_unit = Gq / max(norm(Gq), 1e-9);     % +G(q) 的单位方向（抵消 -G）
            dir_raw = dir_geo + opts.dir_gravity_mix * g_unit;
        else
            dir_raw = dir_geo;
        end
        nd = norm(dir_raw); if nd>0, dir_raw = dir_raw/nd; end
        tau_des = opts.k_dir_max * dir_raw;        % 扭矩种子

        % 5.3 ω 下界、未限功率
        om = dq;
        mask = abs(om) < opts.omega_eps;
        om(mask) = opts.omega_eps .* sign(om(mask)) + opts.omega_eps .* (om(mask)==0);
        p_des = tau_des .* om;

        % 5.4 充满功率（小速度阶段把总功率推近 Pmax）
        if opts.fill_power
            s_now = sum(abs(p_des));
            if s_now > 0 && s_now < opts.fill_alpha * Pmax
                scale = min(opts.p_boost_max, Pmax / s_now);
                p_des = p_des * scale;
            end
        end

        % 5.5 计算/覆盖权重 w
        w_now = predict_w_v2(net, muX, sX, muY, sY, p25, q, dq, qT, m, L, g, ...
                             dlnH, muHv, sHv, opts.useGPU, opts.w_floor);
        % 自举（前 N 步覆盖 NN 权重）
        if k <= opts.bootstrap_steps
            switch lower(opts.bootstrap_mode)
                case 'ratio'
                    s = sum(abs(p_des)); 
                    if s>0, w_now = (abs(p_des(:)).'/s); else, w_now = [1 1 1]/3; end
                case 'equal'
                    w_now = [1 1 1]/3;
            end
        end
        % 肩轴最小占比 & 单纯形重投影
        w_now(1) = max(w_now(1), opts.shoulder_min);
        w_now = project_to_simplex_floor(w_now, opts.w_floor);
        log.w(k,:) = w_now;

        % 5.6 先总功率限幅，再分轴硬帽（双顶帽）
        s_tot = sum(abs(p_des));
        Pmax_eff = Pmax;
        if isa(opts.Pmax_gate,'function_handle')
            Pmax_eff = max(0, min(Pmax, opts.Pmax_gate(norm([xT-xE;yT-yE]), Pmax, k, tvec(k))));
        end
        if s_tot > Pmax_eff && s_tot>0
            p_mid = p_des * (Pmax_eff / s_tot);
        else
            p_mid = p_des;
        end
        cap_i = min(w_now(:)*Pmax_eff, Prated(:));
        p_cap = sign(p_mid) .* min(abs(p_mid), cap_i);

        % 5.7 功率→力矩
        tau_cmd = p_cap ./ om;

        % 5.8 方向一致性保护（绝不朝反）
        if opts.sign_guard
            if (tau_cmd.' * dir_geo) < 0      % 与几何方向夹角>90°
                tau_cmd = -tau_cmd;
                p_cap   = -p_cap;
            end
        end

        % 日志
        log.pow_des(k,:) = p_mid.'; 
        log.pow_lim(k,:) = p_cap.'; 
        log.p(k,:)       = log.pow_lim(k,:);
        log.Pmax(k)      = Pmax; 
        log.Pmax_eff(k)  = Pmax_eff;

        % 5.9 动力学推进（含 OU 阻尼）
        ou_xi = ou_xi + (-ou_xi/0.30)*opts.dt + sqrt(2*opts.dt/0.30)*randn(3,1); % 固定 0.30，足够
        D = diag(max(0, dampNom(:) .* (1 + zetaNom(:) .* ou_xi)));
        ddq = (Mq + 1e-6*eye(3)) \ (tau_cmd - Cq*dq - Gq - D*dq);

        dq = dq + ddq*opts.dt;
        q  = q  + dq *opts.dt;

        % 末端速度与命中
        vE = jacobian_planar(q,L) * dq; log.v(k) = norm(vE);
        log.q(k,:) = q.'; log.dq(k,:) = dq.';
        [xE,yE] = fk_end(q,L); dEE = hypot(xE-xT, yE-yT); log.dEE(k) = dEE;
        if dEE <= opts.radius
            log.reached(k) = true; log.hit = true; t_hit = tvec(k);
            if opts.verbose, fprintf('Hit: t=%.4f s\n', t_hit); end
            return;
        end
    end

    % 未命中
    t_hit = NaN;
    if opts.verbose, fprintf('No hit. Increase t_final or tune dir_gravity_mix/bootstrap.\n'); end
end

% ================= 工具函数 =================
function w = predict_w_v2(net, muX, sX, muY, sY, p25, q, dq, qT, m, L, g, dlnH, muHv, sHv, useGPU, w_floor)
    % 动态特征 F10（按推理口径）
    F10 = build_features_v2(q, dq, qT, m, L, g, dlnH, muHv, sHv, useGPU);
    X   = [p25, F10];
    Xz  = (X - muX) ./ sX;
    yp  = predict(net, Xz);
    yp  = yp .* sY + muY;
    w   = project_to_simplex_floor(yp(1:3), w_floor);
end

function F10 = build_features_v2(q, dq, qT, m, L, g, dlnH, muHv, sHv, useGPU)
    [Mq,~,~] = computeDynamics(m(1),m(2),m(3), L(1),L(2),L(3), g, q, dq);
    p = Mq*dq;
    [xE,yE] = fk_end(q,  L);
    [xT,yT] = fk_end(qT, L);
    dEE = hypot(xE-xT, yE-yT);
    rc = rcond(Mq); condM = max(1, 1/max(rc,1e-15));

    qv = dlarray(single(q(:)),'CB'); pv = dlarray(single(p(:)),'CB');
    if useGPU, qv=gpuArray(qv); pv=gpuArray(pv); end
    [H0,dHdq0,dHdp0] = dlfeval(@hnn_grad_fun, dlnH, qv, pv, muHv, sHv);
    [Hg,~,~] = dlfeval(@hnn_grad_fun, dlnH, ...
                       dlarray(single(qT(:)),'CB'), dlarray(single(zeros(3,1)),'CB'), ...
                       muHv, sHv);
    H0=double(gather(extractdata(H0))); Hg=double(gather(extractdata(Hg)));
    dHdp_norm = double(gather(norm(extractdata(dHdp0))));
    dHdq_norm = double(gather(norm(extractdata(dHdq0))));
    DeltaH = H0 - Hg;

    F10 = [ dEE, diag(Mq).', condM, H0, Hg, DeltaH, dHdp_norm, dHdq_norm ];
end

function [H,dHdq,dHdp] = hnn_grad_fun(dlnet, qv, pv, muHv, sHv)
    z  = ([qv; pv] - muHv) ./ sHv;
    H  = forward(dlnet, z);
    Hs = sum(H,'all');
    [dHdq,dHdp] = dlgradient(Hs, qv, pv);
end

function [x3,y3] = fk_end(q,L)
    q1=q(1); q2=q(2); q3=q(3);
    x1=L(1)*cos(q1);            y1=L(1)*sin(q1);
    x2=x1+L(2)*cos(q1+q2);      y2=y1+L(2)*sin(q1+q2);
    x3=x2+L(3)*cos(q1+q2+q3);   y3=y2+L(3)*sin(q1+q2+q3);
end

function J = jacobian_planar(q,L)
    q1=q(1); q2=q(2); q3=q(3);
    J = [ -L(1)*sin(q1)-L(2)*sin(q1+q2)-L(3)*sin(q1+q2+q3),  -L(2)*sin(q1+q2)-L(3)*sin(q1+q2+q3),  -L(3)*sin(q1+q2+q3);
           L(1)*cos(q1)+L(2)*cos(q1+q2)+L(3)*cos(q1+q2+q3),   L(2)*cos(q1+q2)+L(3)*cos(q1+q2+q3),    L(3)*cos(q1+q2+q3) ];
end

function w = project_to_simplex_floor(w, floor_val)
    w = double(w(:).'); floor_val = max(0, min(1/3 - 1e-6, floor_val));
    free = 1 - 3*floor_val;
    w = max(w,0); s=sum(w); if s==0, w=ones(size(w))/numel(w); else, w=w/s; end
    w = max(w - floor_val, 0); s2=sum(w); if s2>0, w=w/s2*free; end
    w = w + floor_val;
end

function M = merge_opts(M,U)
    if nargin<2 || ~isstruct(U), return; end
    f=fieldnames(U); for i=1:numel(f), M.(f{i}) = U.(f{i}); end
end
