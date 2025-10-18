function spinn_demo_animation_MA()
% 一键动画（PID 物理核 + 双顶帽功率限幅）
% 依赖：spinn_MechanicAlarm.m / computeDynamics.m 已在路径上
% - 若 spinn_MechanicAlarm 支持 gate/ptt/joint/Prated，将自动启用；
% - 若不支持，这些 cfg 字段会被忽略（保持后向兼容）。

    %% ===== 统一设定区（28维 params + cfg + L1/L2/L3）=====
    % 28 维： [m1 m2 m3, dq0(3), damping(3), target(3,deg), init(3,deg), Pmax, ...
    %          Kp1 Ki1 Kd1, Kp2 Ki2 Kd2, Kp3 Ki3 Kd3, w1 w2 w3]
    m1=0.18; m2=0.22; m3=3.40;
    dq0 = [0 0 0];
    damping = [3.2 3.2 3.2];
    tgt_deg  = [35 40 45];          % 目标角(度) —— 可调
    init_deg = [0 0 0];             % 初始角(度) —— 可调
    Pmax = 240;                     % 总功率上限 —— 可调(↑更易命中)
    Kp=[60 60 60]; Ki=[0.20 0.20 0.20]; Kd=[0.10 0.10 0.10];
    w  = [0.34 0.33 0.33];          % 功率份额（分轴硬帽占比）—— 可调
    w  = w / sum(w);                % 规范化

    params = [ m1, m2, m3, dq0, damping, tgt_deg, init_deg, Pmax, ...
               Kp(1), Ki(1), Kd(1), Kp(2), Ki(2), Kd(2), Kp(3), Ki(3), Kd(3), w ];

    % —— 仿真/命中判定配置（spinn_MechanicAlarm 的 cfg）——
    cfg.dt        = 0.002;           % 步长
    cfg.t_final   = 10;             % 时域 —— 若总是不命中，可先调到 5.0 再观察
    cfg.radius    = 0.010;           % 命中半径(米)
    cfg.omega_eps = 1e-3;            % 近零角速度下界（防除零）

    % （可选）启用“近目标收功率”门控 + 电机包络 + 关节硬限位 + 轴额定功率
    % 若你的 spinn_MechanicAlarm 未实现这些选项，下面字段将被忽略，脚本仍可跑。
    cfg.gate = struct('enable',true,'d_on',0.02,'d_off',0.01,'pmin_frac',0.25); % Pmax → Pmax_eff(t)
    cfg.ptt  = struct('omega_floor',0.05,'omega_base',4.0,'use_dq_eff',true);   % p→τ 包络
    cfg.Prated = [110 110 110];                                                  % 轴额定功率硬顶帽
    cfg.joint  = struct('qmin_deg',[-175 5 5], 'qmax_deg',[175 175 175], ...     % 硬限位（不重叠）
                        'deadband_deg',0.5, 'freeze_inward',true, 'zero_vel_on_contact',true);

    % 与物理核保持一致的几何常量
    L1=0.24; L2=0.214; L3=0.324;

    %% ===== 调用仿真内核 =====
    % 老师核（统一物理口径），返回 t_hit 与 info（含功率/轨迹等诊断）
    [t_hit, info] = spinn_MechanicAlarm(params, cfg);

    % --- 提取日志 ---
    t      = info.t(:);
    qh     = info.q_history;                  % k x 3
    Pl     = info.power_lim_history;          % k x 3（总功率+分轴限幅后）
    Ptot   = info.total_power_history(:);
    reached= info.reached;
    v_end  = info.end_speed;
    hasPmaxEff = isfield(info,'Pmax_eff_hist') && numel(info.Pmax_eff_hist)==numel(t);
    if hasPmaxEff, Pmax_eff = info.Pmax_eff_hist(:); end

    % 计算目标末端位置用于画“命中半径”
    tgt = deg2rad(tgt_deg(:));
    [xT,yT] = fk_end(tgt(1), tgt(2), tgt(3), L1, L2, L3);

    % 预计算末端轨迹（动画更稳）
    K = size(qh,1);
    xy1 = zeros(K,2); xy2 = zeros(K,2); xy3 = zeros(K,2);
    for k=1:K
        [x1,y1,x2,y2,x3,y3] = chain_xy(qh(k,:).', L1, L2, L3);
        xy1(k,:)=[x1,y1]; xy2(k,:)=[x2,y2]; xy3(k,:)=[x3,y3];
    end

    %% ===== 画布 =====
    fig = figure('Name','SPINN MechanicAlarm Demo','Color','w','Position',[60 60 1200 560]);
    try, set(fig,'Renderer','opengl'); catch, end

    % 左：机械臂动画
    ax1 = subplot(1,2,1);
    hold(ax1,'on'); axis(ax1,'equal'); grid(ax1,'on');
    R=L1+L2+L3; xlim(ax1,[-R R]); ylim(ax1,[-R R]);
    title(ax1,'3R 机械臂（PID + 双顶帽功率限幅）');

    plot(ax1, xT, yT, 'ro', 'MarkerSize',8, 'LineWidth',1.2);
    hCircle = rectangle(ax1, 'Position',[xT-cfg.radius, yT-cfg.radius, 2*cfg.radius, 2*cfg.radius], ...
                        'Curvature',[1,1], 'EdgeColor',[1 0 0], 'LineStyle','--', 'LineWidth',1.0);
    try, set(hCircle,'EdgeAlpha',0.35); catch, end

    hPath = plot(ax1, xy3(1,1), xy3(1,2), '-', 'LineWidth',1, 'Color',[0.2 0.6 1.0]);
    hArm  = plot(ax1, [0, xy1(1,1), xy2(1,1), xy3(1,1)], ...
                       [0, xy1(1,2), xy2(1,2), xy3(1,2)], ...
                       '-o', 'LineWidth',3, 'MarkerFaceColor',[0 0 0], 'Color',[0.1 0.1 0.1]);
    hEE   = plot(ax1, xy3(1,1), xy3(1,2), 'o', 'MarkerSize',6, 'MarkerFaceColor',[0.2 0.6 1.0], 'Color','k');
    hTxt  = text(ax1, 0.02, 0.98, '', 'Units','normalized','HorizontalAlignment','left', ...
                 'VerticalAlignment','top','FontSize',10,'Color',[0.1 0.1 0.1]);

    % 右上：三轴功率（已限幅）
    ax2 = subplot(2,2,2); hold(ax2,'on'); grid(ax2,'on');
    plot(ax2, t, abs(Pl(:,1)), '-', 'LineWidth',1.2);
    plot(ax2, t, abs(Pl(:,2)), '-', 'LineWidth',1.2);
    plot(ax2, t, abs(Pl(:,3)), '-', 'LineWidth',1.2);
    ylabel(ax2,'|p_i(t)| / W'); title(ax2,'三轴功率（限幅后）');
    legend(ax2,'p_1','p_2','p_3','Location','northeast');
    hCursor2 = xline(ax2, t(1), '--k');

    % 右下：总功率 Σ|p_i|（若有 Pmax_eff(t) 则叠加参考线）
    ax3 = subplot(2,2,4); hold(ax3,'on'); grid(ax3,'on');
    plot(ax3, t, Ptot, 'LineWidth',1.6);
    if hasPmaxEff
        plot(ax3, t, Pmax_eff, ':', 'LineWidth',1.2);      % 动态“收功率”参考线
        legend(ax3,'\Sigma|p_i|','P_{max}^{eff}(t)','Location','best');
    end
    yline(ax3, Pmax, ':r', 'P_{max}');
    xlabel(ax3,'t / s'); ylabel(ax3,'\Sigma |p_i| / W'); title(ax3,'总功率（限幅后）');
    hCursor3 = xline(ax3, t(1), '--k');

    %% ===== 动画主循环 =====
    playback = 1.0;
    dt = mean(diff(t)); if ~isfinite(dt) || dt<=0, dt = cfg.dt; end
    stride = max(1, round(0.004 / dt));
    for k = 1:stride:K
        if ~ishghandle(fig), return; end
        set(hArm, 'XData',[0, xy1(k,1), xy2(k,1), xy3(k,1)], ...
                  'YData',[0, xy1(k,2), xy2(k,2), xy3(k,2)]);
        set(hEE,  'XData',xy3(k,1), 'YData',xy3(k,2));
        set(hPath,'XData',xy3(1:k,1), 'YData',xy3(1:k,2));
        set(hCursor2, 'Value', t(k));
        set(hCursor3, 'Value', t(k));

        % sat_ratio_total / sat_ratio_axis 为老师核诊断字段（新旧版本均有）
        sTot = 100*getfield_safe(info,'sat_ratio_total',0); %#ok<GFLD>
        sAx  = getfield_safe(info,'sat_ratio_axis',[0 0 0]);

        txt = sprintf(['t = %.3f s | reached = %d | v_end = %.3f m/s\n' ...
                       'sat_total = %.1f%% | sat_axis = [%.1f %.1f %.1f]%%\n' ...
                       'w = [%.2f %.2f %.2f] | P_{max}=%g W'], ...
                      t(k), reached, v_end, ...
                      sTot, 100*sAx(1), 100*sAx(2), 100*sAx(3), ...
                      w(1), w(2), w(3), Pmax);
        set(hTxt, 'String', txt);

        drawnow;
        pause((dt*stride)/max(1e-6,playback));
    end

    %% ===== 结果打印 =====
    if ~reached
        warning('本次未命中（time_to_reach = NaN）。建议上调 cfg.t_final 或 cfg.radius，或增加 Pmax / 调整 w。');
    else
        fprintf('命中：time_to_reach = %.3f s，命中瞬时末端速度 = %.3f m/s\n', t_hit, v_end);
    end
end

% ======= 工具函数 =======
function [x1,y1,x2,y2,x3,y3] = chain_xy(q, L1, L2, L3)
    q1=q(1); q2=q(2); q3=q(3);
    x1 = L1*cos(q1);                 y1 = L1*sin(q1);
    x2 = x1 + L2*cos(q1+q2);         y2 = y1 + L2*sin(q1+q2);
    x3 = x2 + L3*cos(q1+q2+q3);      y3 = y2 + L3*sin(q1+q2+q3);
end

function [x3,y3] = fk_end(q1,q2,q3,L1,L2,L3)
    x1 = L1*cos(q1);            y1 = L1*sin(q1);
    x2 = x1 + L2*cos(q1+q2);    y2 = y1 + L2*sin(q1+q2);
    x3 = x2 + L3*cos(q1+q2+q3); y3 = y2 + L3*sin(q1+q2+q3);
end

function val = getfield_safe(S, name, defaultVal)
    if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
        val = S.(name);
    else
        val = defaultVal;
    end
end
