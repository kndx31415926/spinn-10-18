function [t_hit, v_hit, w0, log] = spinn_mechanical_armNN(params25, opts)
% 学生链（带硬限位版）：NN 预测 w + 几何最速方向 + 双顶帽功率约束 + 关节硬限位
%
% 输入
%   params25 : 1x25（与数据生成/训练严格对齐）
%              [m1 m2 m3, dq0_1 dq0_2 dq0_3, damp1 zeta1 damp2 zeta2 damp3 zeta3,
%               init_deg1..3, tgt_deg1..3, dtheta1..3, Pmax, Prated1..3]
%   opts     : 结构体（仅覆写你想改的字段）
%              关键字段：dt,t_final,radius,omega_eps,ou_tau,k_dir_max,
%                        use_pid,pid.{Kp,Ki,Kd},
%                        online_recompute_w,w_floor,fill_power,fill_alpha,p_boost_max,
%                        force_zero_init,
%                        joint  ← ★硬限位：struct('qmin_deg',1x3,'qmax_deg',1x3,
%                                                   'deadband_deg',标量/1x3,
%                                                   'zero_vel_on_contact',bool,
%                                                   'freeze_inward',bool)
%
% 输出
%   t_hit, v_hit : 命中时间（s）与命中瞬时末端速度（m/s），未命中为 NaN
%   w0           : t=0 的 NN 份额（投影 + 下界后）
%   log          : 记录（t,p(3xK),v(1xK),q(3xK),dq(3xK),Pmax,Prated,hit,radius,w(Kx3)）

    if nargin<2, opts=struct(); end
    o = defaults(); fn = fieldnames(o);
    for i=1:numel(fn), if ~isfield(opts,fn{i}), opts.(fn{i}) = o.(fn{i}); end, end

    % ---- 载入模型（优先 SPINN 版；若缺失则回退为等分权重）----
    hasModel = true;
    modelFile = 'trained_model_spinn.mat';
    if ~isfile(modelFile), modelFile = 'trained_model.mat'; end
    if isfile(modelFile)
        S = load(modelFile);    % 需要含 trainedNet, muX, sigmaX, muY, sigmaY
        if ~isfield(S,'trainedNet') || ~isfield(S,'muX') || ~isfield(S,'sigmaX') ...
           || ~isfield(S,'muY') || ~isfield(S,'sigmaY')
            hasModel = false;
            warning('[spinn_mechanical_armNN] 模型文件字段不全，将回退为均分 w。');
        end
    else
        hasModel = false;
        warning('[spinn_mechanical_armNN] 未找到模型文件，将回退为均分 w。');
        S = struct();
    end

    % ---- 解包 25 维参数 ----
    m  = params25(1:3);
    dq = params25(4:6);
    damp = [params25(7) params25(9) params25(11)];
    zeta = [params25(8) params25(10) params25(12)];
    q0_deg = params25(13:15);
    qT_deg = params25(16:18);
    Pmax   = params25(22);
    Prated = params25(23:25);

    if opts.force_zero_init
        dq = [0 0 0]; q0_deg = [0 0 0];
    end

    % ---- 常量、初值 ----
    g=9.81; L=[0.24,0.214,0.324];
    q  = deg2rad(q0_deg(:));
    dq = dq(:);
    qT = deg2rad(qT_deg(:));

    % t 轴
    K = floor(opts.t_final/opts.dt)+1;
    t = (0:K-1)*opts.dt;

    % 记录
    p_hist = zeros(3,K);
    v_hist = zeros(1,K);
    q_hist = zeros(3,K); q_hist(:,1)=q;
    dq_hist= zeros(3,K); dq_hist(:,1)=dq;
    w_hist = zeros(K,3);

    % 先算一次性 w0（用于显示）
    w0 = predict_w(q, dq);

    % OU 噪声 & PID 积分态（仅在 use_pid=true 时使用）
    xi=zeros(3,1); eI=zeros(3,1); ePrev=zeros(3,1);

    hit=false; t_hit=NaN; v_hit=NaN;

    % ===== 主循环 =====
    for k=2:K
        % --- 方向选择：几何最速 或 PID ---
        if ~opts.use_pid
            [xE,yE] = fk_end(q,L); [xT,yT] = fk_end(qT,L);
            r  = [xT-xE; yT-yE]; nr = norm(r); if nr>0, r=r/nr; else, r=[0;0]; end
            J  = jacobian_planar(q,L);
            tau_des = J.' * r;
            tau_des = opts.k_dir_max * tau_des / max(1.0, norm(tau_des));
        else
            e = qT - q; eI = eI + e*opts.dt; eD = (e - ePrev)/opts.dt; ePrev = e;
            tau_des = opts.pid.Kp(:).*e + opts.pid.Ki(:).*eI + opts.pid.Kd(:).*eD;
        end

        % --- 角速度下界（防除零） ---
        om = dq;
        mask = abs(om) < opts.omega_eps;
        om(mask) = opts.omega_eps.*sign(om(mask)) + opts.omega_eps.*(om(mask)==0);

        % --- 期望功率（未限） ---
        p_des = tau_des .* om;                 % p = τ·ω

        % --- 可选：即使未超 Pmax，也把功率“充满”到接近 Pmax ---
        if opts.fill_power
            s_now = sum(abs(p_des));
            if s_now > 0
                scale = min(opts.p_boost_max, Pmax / s_now);
                if s_now < opts.fill_alpha * Pmax
                    p_des = p_des * scale;
                end
            end
        end

        % --- 分轴上限：NN 份额（每步或一次性）+ 额定 Prated ---
        if opts.online_recompute_w
            w_now = predict_w(q, dq);          % 每步在线重算 w(t)
        else
            w_now = w0;                        % 固定一次性 w
        end
        cap_i = min(w_now(:)*Pmax, Prated(:)); % 双顶帽：min(w_i·Pmax, Prated_i)

        % --- 先总功率到 Pmax，再分轴裁剪（双顶帽） ---
        s_tot = sum(abs(p_des));
        if s_tot > Pmax && s_tot>0
            p_des = p_des * (Pmax/s_tot);      % 先“总功率”限幅到 Pmax
        end
        p_cap = sign(p_des) .* min(abs(p_des), cap_i);  % 再“分轴硬帽”
        tau   = p_cap ./ om;

        % --- 动力学推进（与你工程一致） ---
        [Mq, Cq, Gq] = computeDynamics(m(1),m(2),m(3),L(1),L(2),L(3),g, q, dq);  % 现工程通用核。:contentReference[oaicite:3]{index=3}
        xi = xi + (-xi/opts.ou_tau)*opts.dt + sqrt(2*opts.dt/opts.ou_tau)*randn(3,1);
        D  = diag(max(0, damp(:).*(1 + zeta(:).*xi)));
        ddq = Mq \ (tau - Cq*dq - Gq - D*dq);

        dq = dq + ddq*opts.dt;
        q  = q  + dq *opts.dt;

        % ★ 硬限位（可选）：卡边 → 角度钳制 + 置零该轴速度（与老师核一致）
        if isstruct(opts.joint) && ~isempty(opts.joint)
            [q, dq] = enforce_joint_hard_limits(q, dq, opts.joint);
        end

        % 记录
        p_hist(:,k) = tau .* dq;
        q_hist(:,k) = q;
        dq_hist(:,k)= dq;
        w_hist(k,:) = w_now;
        vE = jacobian_planar(q,L)*dq; 
        v_hist(k) = norm(vE);

        % 命中判定
        [xE,yE] = fk_end(q,L); [xT,yT] = fk_end(qT,L);
        if ~hit && hypot(xE-xT,yE-yT) <= opts.radius
            hit=true; t_hit=t(k); v_hit=v_hist(k);
            break;
        end
    end

    k_end = k;
    log = struct('t',t(1:k_end), 'p',p_hist(:,1:k_end), 'v',v_hist(1:k_end), ...
                 'q',q_hist(:,1:k_end), 'dq',dq_hist(:,1:k_end), ...
                 'Pmax',Pmax, 'Prated',Prated, 'hit',hit, ...
                 'radius',opts.radius, 'controller','nn', 'w',w_hist(1:k_end,:));

    % ===== 内嵌：预测 w 并做“下界+投影”（模型缺失时回退为等分） =====
    function w = predict_w(q_now, dq_now)
        if ~hasModel
            w = [1 1 1]/3;
            return;
        end
        try
            init_deg_now = rad2deg(q_now(:)).';
            dth_deg_now  = qT_deg - init_deg_now;
            x = [ params25(1:3), ...                                % m1..3
                  dq_now(:).', ...                                  % dq “当作” dq0
                  params25(7:12), ...                               % damp/zeta
                  init_deg_now, qT_deg, dth_deg_now, ...            % init/tgt/dtheta
                  Pmax, Prated ];                                   % Pmax/Prated
            x_norm = (x - S.muX)./S.sigmaX;
            y_pred = predict(S.trainedNet, x_norm);
            y_pred = y_pred .* S.sigmaY + S.muY;
            w = project_to_simplex_row(y_pred(1:3));
        catch
            % 推理失败则稳健回退
            w = [1 1 1]/3;
        end
        % ★ 关键：给下界，避免 0 份额导致自锁
        if opts.w_floor > 0
            w = max(w, opts.w_floor);
            w = w / sum(w);
        end
    end
end

% ---------- 默认参数 ----------
function o = defaults()
    o.dt = 0.002; o.t_final = 5.0; o.radius = 0.2;
    o.ou_tau = 0.30; o.omega_eps = 1e-3;
    o.k_dir_max = 25;
    o.use_pid = false;
    o.online_recompute_w = true;     % ★ 每步在线重算
    o.force_zero_init = false;

    % 关键新增
    o.w_floor = 0.05;                % NN 输出最小份额（防自锁）
    o.fill_power = true;             % 小速度也“用满”功率
    o.fill_alpha = 0.9;              % 低于 0.9*Pmax 认为“可再放大”
    o.p_boost_max = 5.0;             % 放大上限（防数值爆）

    % 硬限位（默认关闭；与老师核相同字段）
    o.joint = [];                    % 例：struct('qmin_deg',[-175 5 5], 'qmax_deg',[175 175 175], ...
                                     %           'deadband_deg',0.5,'zero_vel_on_contact',true,'freeze_inward',true)

    o.pid = struct('Kp',[60 60 60],'Ki',[0.20 0.20 0.20],'Kd',[0.10 0.10 0.10]);
end

% ---------- 工具 ----------
function w = project_to_simplex_row(w)
    w = max(w(:).',0); s=sum(w);
    if s<=0, w=ones(size(w))/numel(w); else, w=w/s; end
end

function [x3,y3] = fk_end(q,L)
    x1=L(1)*cos(q(1)); y1=L(1)*sin(q(1));
    x2=x1+L(2)*cos(q(1)+q(2)); y2=y1+L(2)*sin(q(1)+q(2));
    x3=x2+L(3)*cos(q(1)+q(2)+q(3)); y3=y2+L(3)*sin(q(1)+q(2)+q(3));
end

function J = jacobian_planar(q,L)
    q1=q(1); q2=q(2); q3=q(3);
    J11=-L(1)*sin(q1)-L(2)*sin(q1+q2)-L(3)*sin(q1+q2+q3);
    J12=-L(2)*sin(q1+q2)-L(3)*sin(q1+q2+q3);
    J13=-L(3)*sin(q1+q2+q3);
    J21= L(1)*cos(q1)+L(2)*cos(q1+q2)+L(3)*cos(q1+q2+q3);
    J22= L(2)*cos(q1+q2)+L(3)*cos(q1+q2+q3);
    J23= L(3)*cos(q1+q2+q3);
    J=[J11 J12 J13; J21 J22 J23];
end

% --- 硬限位：卡边 → 角度钳制 + 置零该轴速度（与老师核一致） ---
function [q_clamp, dq_clamp] = enforce_joint_hard_limits(q, dq, J)
    q_clamp = q; dq_clamp = dq;
    if ~isfield(J,'qmin_deg') || ~isfield(J,'qmax_deg'), return; end
    qmin = deg2rad(J.qmin_deg(:));  qmax = deg2rad(J.qmax_deg(:));
    if numel(qmin)~=3 || numel(qmax)~=3, return; end
    db   = 0.5; if isfield(J,'deadband_deg') && ~isempty(J.deadband_deg), db = J.deadband_deg; end
    db   = deg2rad(db).*ones(3,1);
    lb = qmin + db;  ub = qmax - db;

    z0 = true; if isfield(J,'zero_vel_on_contact'), z0 = logical(J.zero_vel_on_contact); end
    fr = true; if isfield(J,'freeze_inward'),       fr = logical(J.freeze_inward);       end

    for i=1:3
        if q_clamp(i) < lb(i)
            q_clamp(i) = lb(i);
            if z0, dq_clamp(i) = 0; end
            if fr && dq_clamp(i) < 0, dq_clamp(i) = 0; end
        elseif q_clamp(i) > ub(i)
            q_clamp(i) = ub(i);
            if z0, dq_clamp(i) = 0; end
            if fr && dq_clamp(i) > 0, dq_clamp(i) = 0; end
        end
    end
end
