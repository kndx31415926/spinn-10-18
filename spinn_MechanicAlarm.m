function [time_to_reach, info] = spinn_MechanicAlarm(params, cfg)
% SPINN mechanical arm simulator with diagnostics (28-dim -> time [+ info])
%
% 参数向量（1x28，与工程现有布局保持一致）：
% [m1 m2 m3, dq0(3), damping(3), target(3,deg), init(3,deg), Pmax, ...
%  Kp1 Ki1 Kd1, Kp2 Ki2 Kd2, Kp3 Ki3 Kd3, weight1 weight2 weight3]
%
% 可选 cfg 字段（向后兼容；皆可缺省）：
%   .dt        : 步长（默认 0.002）
%   .t_final   : 仿真时长（默认 3.0）
%   .radius    : 命中半径（默认 0.2）
%   .omega_eps : 角速度下界，仅用于数值上避免除零（默认 1e-3）
%   .Prated    : 1x3（W），每轴额定功率上限；缺省则仅用 w·Pmax
%   .joint     : 结构体，硬限位配置（见 enforce_joint_hard_limits 内注释）
%   .tau_rated : 1x3（N·m），每轴扭矩上限（可选；缺省不生效）
%
% 返回：
%   time_to_reach : 首次进入半径的时间(s)，未命中返回 NaN
%   info :
%     .t, .q_history, .dq_history
%     .power_raw_history（限幅前 |tau_pid .* dq|）
%     .power_lim_history（限幅后 |tau .* dq|）
%     .total_power_history, .sat_total_hist, .sat_axis_hist
%     .tau_history, .Pcap_axis, .w, .Prated（若有）, .radius
%     .end_speed, .energy_abs, .reached
%     .joint_clamp_hist(kx3 logical)

    % -------- 参数检查 --------
    if numel(params) ~= 28
        error('spinn_MechanicAlarm: params must be 1x28.');
    end

    % -------- 默认超参 --------
    def.dt        = 0.002;
    def.t_final   = 3.0;
    def.radius    = 0.2;
    def.omega_eps = 1e-3;
    def.Prated    = [];
    def.joint     = [];
    def.tau_rated = [];   % 可选：每轴扭矩上限（N·m）

    % 合并 spinn_defaults（若存在），再被 cfg 覆盖
    if exist('spinn_defaults','file') == 2
        try
            def = merge_struct(def, spinn_defaults());  % 仅覆盖已有字段
        catch
        end
    end
    if nargin >= 2 && ~isempty(cfg)
        def = merge_struct(def, cfg);
    end

    dt        = def.dt;
    t_final   = def.t_final;
    radius    = def.radius;
    omega_eps = def.omega_eps;
    Prated_in = def.Prated;          % [] 或 1x3
    jointCfg  = def.joint;           % [] 或 struct
    tauRated  = def.tau_rated;       % [] 或 1x3

    % -------- 解包（与工程现有口径对齐）--------
    m1 = params(1);  m2 = params(2);  m3 = params(3);
    dq = [params(4); params(5); params(6)];                % 初始 dq
    D  = diag([params(7), params(8), params(9)]);          % 阻尼（名义）
    tgt = deg2rad([params(10); params(11); params(12)]);   % 目标角(rad)
    q   = deg2rad([params(13); params(14); params(15)]);   % 初始角(rad)
    Pmax = params(16);                                     % 总功率上限
    % 注意：Kp/Ki/Kd 的索引是逐轴打包（与数据生成一致）
    Kp = [params(17) params(20) params(23)];
    Ki = [params(18) params(21) params(24)];
    Kd = [params(19) params(22) params(25)];
    w  = [params(26); params(27); params(28)];             % 功率份额（分轴上限）

    % 常量（与工程一致）
    L1 = 0.24; L2 = 0.214; L3 = 0.324;  g = 9.81;

    % 分轴功率上限（双顶帽：min(w·Pmax, Prated)）
    Pcap_axis = (w(:) * Pmax).';                 % 1x3
    if ~isempty(Prated_in)
        if ~isvector(Prated_in) || numel(Prated_in)~=3
            error('cfg.Prated 必须为 1x3 向量（单位：W）。');
        end
        Pcap_axis = min(Pcap_axis, Prated_in(:).');   % 双顶帽
    end

    % 时间网格与记录
    tvec   = 0:dt:t_final;  nT = numel(tvec);
    q_hist = nan(nT,3); dq_hist = nan(nT,3);
    Praw_h = zeros(nT,3);   Plim_h = zeros(nT,3);
    Ptot_h = zeros(nT,1);
    satTot = false(nT,1);   satAx   = false(nT,3);
    tau_h  = zeros(nT,3);
    jclamp = false(nT,3);

    % PID 状态
    e_sum = zeros(3,1); e_prev = zeros(3,1);

    % 目标 EE 坐标
    [x3_t, y3_t] = fk_end(tgt(1), tgt(2), tgt(3), L1, L2, L3);

    % 记录初值
    q_hist(1,:)  = q.';  dq_hist(1,:) = dq.';

    % -------- 起始命中（t=0）预检查 --------
    [x0,y0] = fk_end(q(1),q(2),q(3),L1,L2,L3);
    if hypot(x0 - x3_t, y0 - y3_t) <= radius
        time_to_reach = 0.0;
        reached = true;  kEnd = 1;
        info = build_info(); return;
    end

    % -------- 主循环（关键变更：功率限幅通过“缩放扭矩”，不做 P/ω 回算） --------
    reached = false;
    for k = 2:nT
        % --- PID 扭矩 ---
        e  = tgt - q;
        e_sum = e_sum + e*dt;
        de = (e - e_prev)/dt;  e_prev = e;
        tau_pid = [Kp(1)*e(1) + Ki(1)*e_sum(1) + Kd(1)*de(1);
                   Kp(2)*e(2) + Ki(2)*e_sum(2) + Kd(2)*de(2);
                   Kp(3)*e(3) + Ki(3)*e_sum(3) + Kd(3)*de(3)];

        % ---- 1) 计算“限幅前”功率并记录 ----
        p_raw = abs(tau_pid(:) .* dq(:)).';     % 1x3，逐轴 |tau .* dq|
        Praw_h(k,:) = p_raw;
        P_total_raw = sum(p_raw);

        % ---- 2) 先按“分轴功率上限”裁剪扭矩（不做除法放大）----
        tau_lim = tau_pid(:).';
        for i = 1:3
            if p_raw(i) > (Pcap_axis(i) + 1e-12)
                % 将该轴扭矩按比例压到功率上限
                scale_i = Pcap_axis(i) / (p_raw(i) + eps);
                tau_lim(i) = tau_lim(i) * scale_i;
                satAx(k,i) = true;
            end
        end

        % ---- 3) 再按“总功率上限”整体等比缩放 ----
        p_ax = abs(tau_lim .* dq(:).');      % 轴功率（经分轴裁剪）
        P_after_axis = sum(p_ax);
        if P_after_axis > Pmax && P_after_axis > 0
            s_tot = Pmax / P_after_axis;
            tau_lim = tau_lim * s_tot;
            satTot(k) = true;
        end

        % ---- 4) 可选：扭矩上限（若 cfg.tau_rated 提供则启用）----
        if ~isempty(tauRated)
            if ~isvector(tauRated) || numel(tauRated) ~= 3
                error('cfg.tau_rated 必须为 1x3（单位：N·m）。');
            end
            tau_lim = sign(tau_lim) .* min(abs(tau_lim), tauRated(:).');
        end

        % ---- 5) 推进动力学（半隐式 Euler）----
        [Mq, Cq, Gq] = computeDynamics(m1,m2,m3,L1,L2,L3,g,q,dq);
        ddq = (Mq + 1e-6*eye(3)) \ (tau_lim(:) - Cq*dq - Gq - D*dq);

        dq = dq + ddq*dt;
        q  = q  + dq *dt;

        % ---- 6) 硬限位（可选）：后处理角度/速度 ----
        if isstruct(jointCfg) && ~isempty(jointCfg)
            [q, dq, jmask] = enforce_joint_hard_limits(q, dq, jointCfg);
            jclamp(k,:) = jmask;
        end

        % ---- 记录 ----
        q_hist(k,:)  = q.';    dq_hist(k,:) = dq.';
        Plim_h(k,:)  = abs(tau_lim .* dq(:).');   % 限幅后逐轴功率
        Ptot_h(k)    = sum(Plim_h(k,:));
        tau_h(k,:)   = tau_lim;

        % ---- 命中检查 ----
        [x3,y3] = fk_end(q(1),q(2),q(3),L1,L2,L3);
        if ~reached && hypot(x3 - x3_t, y3 - y3_t) <= radius
            time_to_reach = tvec(k);
            reached = true;  kEnd = k;
            break;
        end
    end

    if ~reached
        time_to_reach = NaN;  kEnd = nT;
    end

    % -------- 汇总 info --------
    info = build_info();

    % ======== 内嵌工具 ========
    function S = build_info()
        S = struct();
        S.t        = tvec(1:kEnd);
        S.q_history  = q_hist(1:kEnd,:);
        S.dq_history = dq_hist(1:kEnd,:);
        S.power_raw_history = Praw_h(1:kEnd,:);
        S.power_lim_history = Plim_h(1:kEnd,:);
        S.total_power_history = Ptot_h(1:kEnd);
        S.sat_total_hist = satTot(1:kEnd);
        S.sat_axis_hist  = satAx(1:kEnd,:);
        S.tau_history    = tau_h(1:kEnd,:);
        S.Pcap_axis      = Pcap_axis;
        S.w              = w(:).';
        if ~isempty(Prated_in), S.Prated = Prated_in(:).'; end
        S.radius = radius;
        % 末端速度与能量
        Jk = jacobian_3R(S.q_history(end,:).', L1,L2,L3);
        S.end_speed = norm(Jk * S.dq_history(end,:).');
        % |功|的时间积分（近似）
        S.energy_abs = trapz(S.t, S.total_power_history);
        S.reached = reached;
        S.joint_clamp_hist = jclamp(1:kEnd,:);
    end
end

% ======================= 几何/工具函数 =======================

function [x3,y3] = fk_end(q1,q2,q3, L1,L2,L3)
    c1 = cos(q1); s1 = sin(q1);
    c12 = cos(q1+q2); s12 = sin(q1+q2);
    c123 = cos(q1+q2+q3); s123 = sin(q1+q2+q3);
    x3 = L1*c1 + L2*c12 + L3*c123;
    y3 = L1*s1 + L2*s12 + L3*s123;
end

function J = jacobian_3R(q, L1,L2,L3)
    q1=q(1); q2=q(2); q3=q(3);
    s1 = sin(q1); c1 = cos(q1);
    s12 = sin(q1+q2); c12 = cos(q1+q2);
    s123 = sin(q1+q2+q3); c123 = cos(q1+q2+q3);
    % 2x3 Jacobian（平面末端）
    J = [ -L1*s1 - L2*s12 - L3*s123,   -L2*s12 - L3*s123,   -L3*s123;
           L1*c1 + L2*c12 + L3*c123,    L2*c12 + L3*c123,    L3*c123 ];
end

function [q_out, dq_out, jmask] = enforce_joint_hard_limits(q, dq, C)
    % C.qmin_deg / C.qmax_deg（1x3）必填；其余可选：
    % C.deadband_deg（默认 0.5），C.zero_vel_on_contact（默认 true）
    qmin = deg2rad(getfield_def(C,'qmin_deg',[-175 -175 -175]));
    qmax = deg2rad(getfield_def(C,'qmax_deg',[ 175  175  175]));
    dead = deg2rad(getfield_def(C,'deadband_deg',0.5));
    zv   = getfield_def(C,'zero_vel_on_contact',true);

    q_out = q; dq_out = dq;
    jmask = false(1,3);

    for i=1:3
        if q_out(i) < (qmin(i) + dead)
            q_out(i) = qmin(i);
            if zv && dq_out(i) < 0, dq_out(i) = 0; end
            jmask(i) = true;
        elseif q_out(i) > (qmax(i) - dead)
            q_out(i) = qmax(i);
            if zv && dq_out(i) > 0, dq_out(i) = 0; end
            jmask(i) = true;
        end
    end
end

function v = getfield_def(S, name, defv)
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        v = S.(name);
    else
        v = defv;
    end
end

function O = merge_struct(O, U)
    if isempty(U) || ~isstruct(U), return; end
    f = fieldnames(U);
    for i=1:numel(f)
        O.(f{i}) = U.(f{i});
    end
end
