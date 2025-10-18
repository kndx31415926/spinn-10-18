function [t_hit, v_hit, log] = spinn_mechanical_arm_pid(params25, w, opts)
% 基线链：PID + 双顶帽功率约束 + 可选“关节硬限位”
% 约束顺序：先总功率 Pmax，再分轴 min(w_i*Pmax, Prated_i)
% 新增：opts.joint 与老师核一致的硬限位接口：
%   opts.joint = struct('qmin_deg',1x3,'qmax_deg',1x3, ...
%                       'deadband_deg',标量/1x3, ...
%                       'zero_vel_on_contact',bool, ...
%                       'freeze_inward',bool);

    if nargin < 3, opts = struct(); end
    if nargin < 2 || isempty(w), w = [1 1 1]/3; end
    w = project_to_simplex_row(w);

    % -------- 默认参数合并 --------
    opts0 = spinn_defaults_local();
    fns = fieldnames(opts0);
    for i=1:numel(fns)
        if ~isfield(opts,fns{i}) || isempty(opts.(fns{i}))
            opts.(fns{i}) = opts0.(fns{i});
        end
    end

    % -------- 取出 25 维参数（新 schema）--------
    % [m1 m2 m3, dq0(3), damp1 zeta1 damp2 zeta2 damp3 zeta3,
    %  init_deg1..3, tgt_deg1..3, dtheta1..3, Pmax, Prated1..3]
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

    % -------- 常量与初值（与你工程口径一致）--------
    g=9.81; L=[0.24,0.214,0.324];
    q  = deg2rad(q0_deg(:));      % rad
    dq = dq(:);                   % rad/s
    qT = deg2rad(qT_deg(:));

    % 记录区
    K = floor(opts.t_final/opts.dt)+1;
    t = (0:K-1)*opts.dt;
    p_hist = zeros(3,K);
    v_hist = zeros(1,K);
    q_hist = zeros(3,K);  q_hist(:,1)=q;
    dq_hist= zeros(3,K);  dq_hist(:,1)=dq;

    % 状态
    hit=false; t_hit=NaN; v_hit=NaN;
    eI=zeros(3,1); ePrev=zeros(3,1);
    xi=zeros(3,1);                                  % OU 抖动状态（与 NN 链/老师核一致风格）

    % -------- 主循环 --------
    for k=2:K
        % --- PID 期望扭矩 ---
        e = qT - q; eI = eI + e*opts.dt; eD = (e - ePrev)/opts.dt; ePrev = e;
        tau_des = opts.pid.Kp(:).*e + opts.pid.Ki(:).*eI + opts.pid.Kd(:).*eD;

        % --- 双顶帽功率约束（先总功率，再分轴） ---
        om = dq;
        mask = abs(om) < opts.omega_eps;
        om(mask) = opts.omega_eps .* sign(om(mask)) + opts.omega_eps .* (om(mask)==0); % 防除零

        p_unlim = tau_des .* om;

        s_tot = sum(abs(p_unlim));
        if s_tot > Pmax && s_tot>0
            p_unlim = p_unlim * (Pmax/s_tot);      % 先“总功率”限到 Pmax
        end

        cap_i = min(w(:)*Pmax, Prated(:));         % 再“分轴硬帽”
        p_cap = sign(p_unlim) .* min(abs(p_unlim), cap_i);
        tau   = p_cap ./ om;

        % --- 动力学推进（computeDynamics 核） ---
        [Mq, Cq, Gq] = computeDynamics(m(1),m(2),m(3),L(1),L(2),L(3),g, q, dq);  % 物理核一致。:contentReference[oaicite:1]{index=1}
        xi = xi + (-xi/opts.ou_tau)*opts.dt + sqrt(2*opts.dt/opts.ou_tau)*randn(3,1);
        D  = diag(max(0, damp(:).*(1 + zeta(:).*xi)));

        ddq = Mq \ (tau - Cq*dq - Gq - D*dq);
        dq  = dq + ddq*opts.dt;
        q   = q  + dq *opts.dt;

        % ★ 硬限位（可选）：卡边→角度钳制 + 置零该轴速度（接口与老师核一致）
        if isstruct(opts.joint) && ~isempty(opts.joint)
            [q, dq] = enforce_joint_hard_limits(q, dq, opts.joint);
        end

        % 记录
        p_hist(:,k) = tau .* dq;
        q_hist(:,k) = q;
        dq_hist(:,k)= dq;

        vE = jacobian_planar(q,L)*dq; 
        v_hist(k) = norm(vE);

        % 命中判定（与 compare/动画一致口径）
        [xE,yE] = fk_end(q,L); [xT,yT] = fk_end(qT,L);
        if ~hit && hypot(xE-xT,yE-yT) <= opts.radius
            hit = true; t_hit = t(k); v_hit = v_hist(k);
            break;
        end
    end

    k_end = k;
    log = struct('t',t(1:k_end), 'p',p_hist(:,1:k_end), 'v',v_hist(1:k_end), ...
                 'q',q_hist(:,1:k_end), 'dq',dq_hist(:,1:k_end), ...
                 'Pmax',Pmax, 'Prated',Prated, 'hit',hit, ...
                 'radius',opts.radius, 'controller','pid', 'w',w);  % 基线 w 为常数

end

% ---------- 默认参数 ----------
function opts = spinn_defaults_local()
    opts.dt = 0.002; opts.t_final = 5.0; opts.radius = 0.2;
    opts.ou_tau = 0.30; opts.omega_eps = 1e-3;
    opts.force_zero_init = false;
    opts.pid = struct('Kp',[60 60 60],'Ki',[0.20 0.20 0.20],'Kd',[0.10 0.10 0.10]);

    % ★ 新增：硬限位（默认关闭；与老师核/NN 链字段一致）
    opts.joint = [];   % 例：struct('qmin_deg',[-175 5 5],'qmax_deg',[175 175 175], ...
                       %           'deadband_deg',0.5,'zero_vel_on_contact',true,'freeze_inward',true)
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

% --- 关节硬限位：卡边 → 角度钳制 + 置零该轴速度（与老师核一致） ---
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
