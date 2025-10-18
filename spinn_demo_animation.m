function spinn_demo_animation()
% 一键动画（逐帧刷新）— 基于 spinn_MechanicAlarm
% 依赖：spinn_MechanicAlarm.m / computeDynamics.m 已在路径上

    %% 1) 内置参数（改这里就能“看着调范围”）
    m1=0.18; m2=0.22; m3=1.40;            % 质量 kg
    dq0 = [0 0 0];                         % 初始角速度 rad/s
    damp = [3.2 3.2 3.2];                  % 阻尼
    init_deg = [0 0 0];                    % 初始角 deg
    tgt_deg  = [35 40 45];                 % 目标角 deg
    Pmax = 240;                            % 总功率上限 W
    Kp=[60 60 60]; Ki=[0.20 0.20 0.20]; Kd=[0.10 0.10 0.10]; % PID
    w  = [0.34 0.33 0.33]; w = w/sum(w);   % 常数功率份额

    cfg.dt = 0.002;     % 仿真步长
    cfg.t_final = 3.0;  % 最长仿真时间
    cfg.radius  = 0.01;  % 命中半径 (m)
 
    % 28维参数向量（与 spinn_MechanicAlarm 的接口完全一致）
    % [m1 m2 m3, dq0(3), damping(3), target(3,deg), init(3,deg), Pmax,
    %  Kp1 Ki1 Kd1, Kp2 Ki2 Kd2, Kp3 Ki3 Kd3, w1 w2 w3]
    params = [ m1, m2, m3, dq0, damp, tgt_deg, init_deg, Pmax, ...
               Kp(1), Ki(1), Kd(1), Kp(2), Ki(2), Kd(2), Kp(3), Ki(3), Kd(3), w ];

    %% 2) 仿真（你的内核）
    [t_hit, info] = spinn_MechanicAlarm(params, cfg); %#ok<NASGU>   % :contentReference[oaicite:1]{index=1}
    t   = info.t(:);
    qh  = info.q_history;                       % k x 3
    Pl  = info.power_lim_history;               % k x 3（限幅后）
    Ptot= info.total_power_history(:);
    reached = info.reached;

    % 与工程口径一致的几何常量（computeDynamics 同口径） :contentReference[oaicite:2]{index=2}
    L1=0.24; L2=0.214; L3=0.324;

    % 目标末端位置（用于画目标与命中半径）
    tgt = deg2rad(tgt_deg(:));
    [xT, yT] = fk_end(tgt(1), tgt(2), tgt(3), L1, L2, L3);

    % 预计算连杆端点坐标（动画更稳）
    kEnd = size(qh,1);
    xy1 = zeros(kEnd,2); xy2 = zeros(kEnd,2); xy3 = zeros(kEnd,2);
    for k = 1:kEnd
        q = qh(k,:).';
        [x1,y1,x2,y2,x3,y3] = chain_xy(q, L1, L2, L3);
        xy1(k,:)=[x1,y1]; xy2(k,:)=[x2,y2]; xy3(k,:)=[x3,y3];
    end

    %% 3) 画布与对象
    fig = figure('Name','SPINN Arm Animation','Color','w','Position',[60 60 1200 540]);
    try, set(fig,'Renderer','opengl'); catch, end

    % 左：机械臂动画
    ax1 = subplot(1,2,1);
    hold(ax1,'on'); axis(ax1,'equal'); grid(ax1,'on');
    R=L1+L2+L3; xlim(ax1,[-R R]); ylim(ax1,[-R R]);
    title(ax1,'3R 机械臂（PID + 功率限幅）');

    plot(ax1, xT, yT, 'ro', 'MarkerSize',8, 'LineWidth',1.5);     % 目标
    hCircle = rectangle(ax1, 'Position',[xT-cfg.radius, yT-cfg.radius, 2*cfg.radius, 2*cfg.radius], ...
                        'Curvature',[1,1], 'EdgeColor',[1 0 0], 'LineStyle','--', 'LineWidth',1.0);
    try, set(hCircle,'EdgeAlpha',0.35); catch, end                 % 兼容不同版本

    hPath = plot(ax1, xy3(1,1), xy3(1,2), '-', 'LineWidth',1, 'Color',[0.2 0.6 1.0]);
    hArm  = plot(ax1, [0, xy1(1,1), xy2(1,1), xy3(1,1)], ...
                       [0, xy1(1,2), xy2(1,2), xy3(1,2)], ...
                       '-o', 'LineWidth',3, 'MarkerFaceColor',[0 0 0], 'Color',[0.1 0.1 0.1]);
    hEE   = plot(ax1, xy3(1,1), xy3(1,2), 'o', 'MarkerSize',6, 'MarkerFaceColor',[0.2 0.6 1.0], 'Color','k');
    hTxt  = text(ax1, 0.02, 0.98, '', 'Units','normalized','HorizontalAlignment','left', ...
                 'VerticalAlignment','top','FontSize',10,'Color',[0.1 0.1 0.1]);

    % 右上：三轴功率 |p_i(t)|
    ax2 = subplot(2,2,2); hold(ax2,'on'); grid(ax2,'on');
    plot(ax2, t, abs(Pl(:,1)), '-', 'LineWidth',1.2);
    plot(ax2, t, abs(Pl(:,2)), '-', 'LineWidth',1.2);
    plot(ax2, t, abs(Pl(:,3)), '-', 'LineWidth',1.2);
    ylabel(ax2,'|p_i(t)| / W'); title(ax2,'三轴功率（限幅后）'); legend(ax2,'p_1','p_2','p_3','Location','northeast');
    hCursor2 = xline(ax2, t(1), '--k');

    % 右下：总功率 Σ|p_i|
    ax3 = subplot(2,2,4); hold(ax3,'on'); grid(ax3,'on');
    plot(ax3, t, Ptot, 'LineWidth',1.6);
    yline(ax3, Pmax, ':r', 'P_{max}');
    xlabel(ax3,'t / s'); ylabel(ax3,'\Sigma |p_i| / W'); title(ax3,'总功率（限幅后）');
    hCursor3 = xline(ax3, t(1), '--k');

    %% 4) 动画主循环（逐帧刷新 + 轻微 pause）
    playback = 1.0;                               % 1.0=实时
    stride   = max(1, round(0.004 / cfg.dt));     % 刷新帧步
    for k = 1:stride:kEnd
        if ~ishghandle(fig), return; end
        set(hArm, 'XData',[0, xy1(k,1), xy2(k,1), xy3(k,1)], ...
                  'YData',[0, xy1(k,2), xy2(k,2), xy3(k,2)]);
        set(hEE,  'XData',xy3(k,1), 'YData',xy3(k,2));
        set(hPath,'XData',xy3(1:k,1), 'YData',xy3(1:k,2));

        set(hCursor2, 'Value', t(k));
        set(hCursor3, 'Value', t(k));
        % ★ 修正：必须带属性名 'String'
        set(hTxt, 'String', sprintf('t = %.3f s | reached = %d | end-speed = %.3f m/s', ...
                                    t(k), reached, info.end_speed));

        drawnow; 
        pause((cfg.dt*stride)/max(1e-6,playback));
    end
end

% ============== 工具函数（与仿真核口径一致） ==============
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
