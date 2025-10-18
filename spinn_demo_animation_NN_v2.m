function spinn_demo_animation_NN_v2()
% 一键动画（NN v2 推理版，严格对齐 spinn_demo_animation_NN 的风格）
% 修复点：
%   1) 命中后机械臂“消失”：将动画长度裁到命中帧/最后有效帧，避免 NaN 进入绘图。
%   2) 终端输出完全对齐 spinn_demo_animation_NN（命中/未命中文案）。
% 依赖：
%   - spinn_mechanical_armNN_v2.m / computeDynamics.m 在路径上
%   - trained_model_spinn_v2.mat 可读

    %% ===== 1) 参数区（与原脚本一致） =====
    m1=0.18; m2=0.22; m3=3.40;
    dq0 = [0 0 0];
    damping = [3.2 3.2 3.2];
    zeta    = [0.01 0.01 0.01];
    tgt_deg  = [35 40 45];
    init_deg = [0 0 0];
    Pmax   = 60;                   % 总功率上限（W）——与原 NN 脚本一致
    Prated = [110 110 110];        % 每轴额定功率（W）

    % 仅当 opts.use_pid=true 时用到（保持接口一致）
    Kp=[60 60 60]; Ki=[0.20 0.20 0.20]; Kd=[0.10 0.10 0.10];

    % 仿真/命中判定配置（传给 NN 学生链）
    cfg.dt        = 0.002;
    cfg.t_final   = 10;
    cfg.radius    = 0.010;
    cfg.omega_eps = 1e-3;

    % 仅用于几何与绘图
    L1=0.24; L2=0.214; L3=0.324;

    %% ===== 2) 组装 25 维输入（与原脚本相同） =====
    m      = [m1 m2 m3];
    dampZ  = [damping(1) zeta(1) damping(2) zeta(2) damping(3) zeta(3)];
    dth_deg= tgt_deg - init_deg;
    params25 = [ m, dq0, dampZ, init_deg, tgt_deg, dth_deg, Pmax, Prated ];

    model_v2 = 'trained_model_spinn_v2.mat';

    % 学生链选项（对齐原 NN 学生核口径；必要的 v2 选项用默认）
    v2opts = struct( ...
        'dt',cfg.dt,'t_final',cfg.t_final,'radius',cfg.radius,'omega_eps',cfg.omega_eps, ...
        'use_pid', false, ...
        'pid', struct('Kp',Kp,'Ki',Ki,'Kd',Kd), ...
        'online_recompute_w', true, ...
        'w_floor', 0.05, ...
        'fill_power', true, 'fill_alpha', 0.9, 'p_boost_max', 5.0);

    %% ===== 3) 调用 NN v2 学生链，拿日志 =====
    [t_hit, log] = spinn_mechanical_armNN_v2(params25, model_v2, v2opts); %#ok<ASGLU>
    assert(isfield(log,'t') && ~isempty(log.t), '学生链未返回 log.t');

    %% ===== 4) 统一维度 + “有效帧”裁剪（关键修复：不让 NaN 进绘图） =====
    t = log.t(:);                       % Kx1
    qh = ensure_Kx3(getfield_safe(log,'q'));
    dq = ensure_Kx3(getfield_safe(log,'dq'));
    % v2 一般提供 pow_lim；若也有 p，则优先 p（与原版口径一致）
    if isfield(log,'p')
        Pk3 = ensure_Kx3(log.p);
    else
        Pk3 = ensure_Kx3(getfield_safe(log,'pow_lim'));
    end
    Ptot = sum(abs(Pk3), 2);

    % 命中帧（若有）
    hit_mask = false(numel(t),1);
    if isfield(log,'reached') && ~isempty(log.reached)
        hit_mask = logical(log.reached(:));
    elseif isfield(log,'hit') && islogical(log.hit) && log.hit
        % 某些实现只给标量 hit，不给逐帧 reached；此时不裁剪到命中帧，只按有效帧裁剪
        hit_mask = false(numel(t),1);
    end
    if any(hit_mask)
        hit_idx = find(hit_mask,1,'first');
    else
        hit_idx = NaN;
    end

    % 逐帧有效性（q/p 有 NaN 的帧认为无效）
    valid_mask = all(isfinite(qh),2) & all(isfinite(Pk3),2);
    if any(~valid_mask)
        last_valid = find(valid_mask,1,'last');
    else
        last_valid = numel(t);
    end

    % 裁剪长度：若有命中，则裁到命中帧；否则裁到最后一个有效帧
    if ~isnan(hit_idx)
        K = hit_idx;
    else
        K = last_valid;
    end
    K = max(2, K);   % 至少保证两帧
    t   = t(1:K);
    qh  = qh(1:K,:);
    dq  = dq(1:K,:);
    Pk3 = Pk3(1:K,:);
    Ptot= Ptot(1:K);

    % reached 标志：与原脚本保持布尔标量
    reached = ~isnan(hit_idx);

    %% ===== 5) 预计算连杆端点（动画更稳） =====
    xy1 = zeros(K,2); xy2 = zeros(K,2); xy3 = zeros(K,2);
    for k=1:K
        [x1,y1,x2,y2,x3,y3] = chain_xy(qh(k,:).', L1, L2, L3);
        xy1(k,:)=[x1,y1]; xy2(k,:)=[x2,y2]; xy3(k,:)=[x3,y3];
    end
    tgt = deg2rad(tgt_deg(:));
    [xT,yT] = fk_end(tgt(1), tgt(2), tgt(3), L1, L2, L3);

    % 末端速度范数（用于文本和 v_hit 计算）
    v_end = zeros(K,1);
    for k=1:K
        Jk = jacobian_3R(qh(k,:), [L1 L2 L3]);
        v_end(k) = norm(Jk * dq(k,:).');
    end

    %% ===== 6) 画布（与原 NN 脚本一致） =====
    fig = figure('Name','SPINN NN Inference Demo (v2)','Color','w','Position',[60 60 1200 540]);
    try, set(fig,'Renderer','opengl'); catch, end

    % 左：机械臂动画
    ax1 = subplot(1,2,1);
    hold(ax1,'on'); axis(ax1,'equal'); grid(ax1,'on');
    R=L1+L2+L3; xlim(ax1,[-R R]); ylim(ax1,[-R R]);
    title(ax1,'3R 机械臂（NN 推理 + 双顶帽功率限幅）');   % 与原版一致风格  :contentReference[oaicite:1]{index=1}
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

    % 右上：三轴功率（限幅后）
    ax2 = subplot(2,2,2); hold(ax2,'on'); grid(ax2,'on');
    plot(ax2, t, abs(Pk3(:,1)), '-', 'LineWidth',1.2);
    plot(ax2, t, abs(Pk3(:,2)), '-', 'LineWidth',1.2);
    plot(ax2, t, abs(Pk3(:,3)), '-', 'LineWidth',1.2);
    ylabel(ax2,'|p_i(t)| / W'); title(ax2,'三轴功率（限幅后）');
    legend(ax2,'p_1','p_2','p_3','Location','northeast');
    hCursor2 = xline(ax2, t(1), '--k');

    % 右下：总功率 Σ|p_i|
    ax3 = subplot(2,2,4); hold(ax3,'on'); grid(ax3,'on');
    plot(ax3, t, Ptot, 'LineWidth',1.6);
    yline(ax3, Pmax, ':r', 'P_{max}');
    xlabel(ax3,'t / s'); ylabel(ax3,'\Sigma |p_i| / W'); title(ax3,'总功率（限幅后）');
    hCursor3 = xline(ax3, t(1), '--k');

    %% ===== 7) 动画主循环（与原脚本完全一致的抽帧策略） =====
    playback = 1.0;
    dt_est = mean(diff(t)); if ~isfinite(dt_est) || dt_est<=0, dt_est = cfg.dt; end
    stride = max(1, round(0.004 / dt_est));
    v_now = 0;

    % 文本里的 w：若 log.w 有效就用；否则均分
    w0 = [1/3 1/3 1/3];
    if isfield(log,'w') && ~isempty(log.w)
        idx0 = find(all(isfinite(log.w),2),1,'first');
        if ~isempty(idx0), w0 = log.w(idx0,:); end
    end

    for k = 1:stride:K
        if ~ishghandle(fig), return; end

        set(hArm, 'XData',[0, xy1(k,1), xy2(k,1), xy3(k,1)], ...
                  'YData',[0, xy1(k,2), xy2(k,2), xy3(k,2)]);
        set(hEE,  'XData',xy3(k,1), 'YData',xy3(k,2));
        set(hPath,'XData',xy3(1:k,1), 'YData',xy3(1:k,2));
        set(hCursor2, 'Value', t(k));
        set(hCursor3, 'Value', t(k));

        if numel(v_end)>=k && isfinite(v_end(k)), v_now = v_end(k); end
        w_show = w0;
        if isfield(log,'w') && size(log.w,1)>=k && all(isfinite(log.w(k,:)))
            w_show = log.w(k,:);
        end

        txt = sprintf(['t = %.3f s | hit = %d | v_end = %.3f m/s\n' ...
                       'w_{NN} = [%.2f %.2f %.2f] | P_{max}=%g W\n' ...
                       'cfg: dt=%.4f s, T=%.2f s, R=%.3f m, \\omega_{eps}=%.1e'], ...
                      t(k), reached, v_now, ...
                      w_show(1), w_show(2), w_show(3), Pmax, ...
                      cfg.dt, cfg.t_final, cfg.radius, cfg.omega_eps);
        set(hTxt, 'String', txt);

        drawnow;
        pause((dt_est*stride)/max(1e-6,playback));
    end

    %% ===== 8) 结果提示（完全对齐原 NN 脚本的输出） =====
    if ~reached
        warning('本次未命中（time_to_reach = NaN）。建议上调 cfg.t_final 或 cfg.radius，或增加 Pmax。');
        fprintf('未命中：time_to_reach = NaN，末端速度(末帧) = %.3f m/s\n', v_now);
    else
        % 命中时刻与瞬时末端速度（若 t_hit 为空，则以 t(hit_idx) 替代）
        if ~isempty(t_hit) && isfinite(t_hit)
            t_hit_print = t_hit;
        elseif ~isnan(hit_idx)
            t_hit_print = t(hit_idx);
        else
            t_hit_print = t(end);
        end
        v_hit = v_end(min(K, max(1, round((t_hit_print - t(1))/dt_est)+1)));
        fprintf('命中：time_to_reach = %.3f s，命中瞬时末端速度 = %.3f m/s\n', t_hit_print, v_hit);
    end
end

% ======= 工具函数（与原脚本风格一致）=======
function X = getfield_safe(S, name)
    if isstruct(S) && isfield(S, name), X = S.(name); else, X = []; end
end

function M = ensure_Kx3(M)
    if isempty(M), M = nan(1,3); end
    if size(M,2) == 3
        % ok: Kx3
    elseif size(M,1) == 3 && size(M,2) ~= 3
        M = M.';  % 3xK -> Kx3
    else
        % 宽松处理：若既非 Kx3 也非 3xK，尝试 reshape；失败则报错
        if numel(M)==3
            M = reshape(M,1,3);
        else
            error('字段维度异常，期望 Kx3 或 3xK，得到 %dx%d。', size(M,1), size(M,2));
        end
    end
end

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

function J = jacobian_3R(q, L)
    q1=q(1); q2=q(2); q3=q(3);
    s1=sin(q1);      c1=cos(q1);
    s12=sin(q1+q2);  c12=cos(q1+q2);
    s123=sin(q1+q2+q3); c123=cos(q1+q2+q3);
    L1=L(1); L2=L(2); L3=L(3);
    J = [ -L1*s1 - L2*s12 - L3*s123,   -L2*s12 - L3*s123,   -L3*s123;
           L1*c1 + L2*c12 + L3*c123,    L2*c12 + L3*c123,    L3*c123 ];
end
