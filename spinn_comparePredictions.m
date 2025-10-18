function metrics = spinn_comparePredictions(n, data_path, var_name, w_baseline, init_angles_deg)
% 仅对比：PID 基线 vs NN（每步在线重算 w）
% ★物理口径对齐★：两条链均启用相同的“关节硬限位 + 双顶帽功率约束 + computeDynamics 核”
% 签名保持不变：spinn_comparePredictions(n, data_path, var_name, w_baseline, init_angles_deg)
%
% 输出 metrics 中包含：时间/末速/节省百分比、w_baseline 与 NN 初值 w0 等。

    %% ---------- 1) 数据文件与变量名 ----------
    if nargin < 2 || isempty(data_path)
        candidates = { ...
            'C:\Users\kndx9\Desktop\SpinnMechanicalArmParams.mat', ...
            'C:\Users\kndx9\Desktop\MechanicalArmParams_spinn.mat', ...
            'C:\Users\kndx9\Desktop\MechanicalArmParams.mat' };
        data_path = '';
        for i=1:numel(candidates)
            if isfile(candidates{i}), data_path = candidates{i}; break; end
        end
        if isempty(data_path)
            error('未找到默认数据文件，请显式传入 data_path。');
        end
    end
    if nargin < 3 || isempty(var_name), var_name = 'params_matrix_spinn'; end

    S = load(data_path);
    if ~isfield(S, var_name)
        if isfield(S,'params_matrix_spinn'), var_name='params_matrix_spinn';
        elseif isfield(S,'params_matrix'),   var_name='params_matrix';
        else, error('在 %s 中未找到 params_matrix[_spinn]。', data_path);
        end
    end
    PM = S.(var_name);
    if n < 1 || n > size(PM,1), error('n 超出数据行数范围（1..%d）。', size(PM,1)); end
    if size(PM,2) < 25, error('数据列数不足（需要≥25列）。'); end

    %% ---------- 2) 取第 n 行的 25 维输入，并设置初始角 ----------
    params25 = PM(n,1:25);
    if nargin < 4 || isempty(w_baseline), w_baseline = [1 1 1]/3; end
    if nargin < 5 || isempty(init_angles_deg), init_angles_deg = [0 0 0]; end
    params25(4:6)   = 0;                 % dq0_1..3 = 0（与动画脚本口径一致）
    params25(13:15) = init_angles_deg;   % 初始角（度）

    % 基线 w 规范化到单纯形
    w_baseline = project_to_simplex_row(w_baseline);

    %% ---------- 3) 统一物理口径（两条链共享） ----------
    % 仿真/命中判定配置
    cfg_common = struct( ...
        'dt',        0.002, ...   % 步长
        't_final',   3.0,   ...   % 若不命中可临时调到 5.0 再观察
        'radius',    0.010, ...   % 命中半径(米)
        'omega_eps', 1e-3);       % 近零角速度下界（防除零）

    % ★关节“硬限位”统一：三轴一致；deadband 为 0.5°
    joint_cfg = struct('qmin_deg',[-175 5 5], 'qmax_deg',[175 175 175], ...
                       'deadband_deg',0.5, 'zero_vel_on_contact',true, 'freeze_inward',true);

    % PID 链选项（与 NN 链的 computeDynamics / 限幅顺序一致；仅控制律不同）
    opts_pid = struct('dt',cfg_common.dt, 't_final',cfg_common.t_final, 'radius',cfg_common.radius, ...
                      'omega_eps',cfg_common.omega_eps, 'force_zero_init',false, ...
                      'ou_tau',0.30, 'pid',struct('Kp',[60 60 60],'Ki',[0.20 0.20 0.20],'Kd',[0.10 0.10 0.10]), ...
                      'joint', joint_cfg);

    % NN 链选项（几何最速方向 + 每步在线重算 w + 下界防自锁 + 起步“充满功率”）
    opts_nn = struct('dt',cfg_common.dt, 't_final',cfg_common.t_final, 'radius',cfg_common.radius, ...
                     'omega_eps',cfg_common.omega_eps, 'ou_tau',0.30, ...
                     'use_pid',false, 'online_recompute_w',true, 'k_dir_max',25, ...
                     'w_floor',0.05, 'fill_power',true, 'fill_alpha',0.9, 'p_boost_max',5.0, ...
                     'joint', joint_cfg);

    %% ---------- 4) 两条仿真链：PID 基线 & NN（逐步重算 w） ----------
    % PID 基线（常数 w；共用 computeDynamics 与“先总功率→再分轴(含 Prated)”的双顶帽）
    [t0, v0, log0] = spinn_mechanical_arm_pid(params25, w_baseline, opts_pid);

    % NN 学生：逐步在线重算 w，并用几何最速方向
    [t1, v1, w0, log1] = spinn_mechanical_armNN(params25, opts_nn);

    %% ---------- 5) 指标 ----------
    timesaved = (t0 - t1) / max(t0, eps);
    speed_red = (v0 - v1) / max(v0, 1e-9);

    fprintf('PID baseline   : t_hit = %.3f s, v_hit = %.3f m/s\n', t0, v0);
    fprintf('NN (online)    : t_hit = %.3f s, v_hit = %.3f m/s, w0 = [%.2f %.2f %.2f]\n', ...
            t1, v1, w0(1), w0(2), w0(3));
    fprintf('Time saved: %.1f%%,  Speed at hit ↓ %.1f%%\n', 100*timesaved, 100*speed_red);

    %% ---------- 6) 画图 ----------
    % 速度曲线
    figure('Name', sprintf('速度对比(第 %d 行)', n)); hold on;
    plot(log0.t, log0.v, '-',  'LineWidth', 2, 'DisplayName', 'PID baseline');
    plot(log1.t, log1.v, '--', 'LineWidth', 2, 'DisplayName', 'NN (online)');
    xlabel('时间 (s)'); ylabel('末端速度 (m/s)');
    title(sprintf('速度对比（第 %d 行样本）', n));
    legend('show'); grid on;

    % 功率曲线（逐轴 + 总功率）
    figure('Name', sprintf('功率对比(第 %d 行)', n));
    for ax = 1:3
        subplot(4,1,ax);
        plot(log0.t, log0.p(ax,:), '-',  'LineWidth', 1.5); hold on;
        plot(log1.t, log1.p(ax,:), '--', 'LineWidth', 1.5);
        ylabel('W'); title(sprintf('轴%d功率', ax)); grid on;
        if ax==1, legend('PID','NN-online'); end
    end
    subplot(4,1,4);
    plot(log0.t, sum(abs(log0.p),1), '-',  'LineWidth', 1.5); hold on;
    plot(log1.t, sum(abs(log1.p),1), '--', 'LineWidth', 1.5);
    xlabel('时间 (s)'); ylabel('W'); title('总功率 |p_1|+|p_2|+|p_3|'); grid on; legend('PID','NN-online');

    % 可选：w(t) 变化（NN 在线）
    figure('Name', sprintf('NN 在线 w(t) - 第 %d 行', n));
    plot(log1.t, log1.w(:,1), '-', 'LineWidth', 1.5); hold on;
    plot(log1.t, log1.w(:,2), '-', 'LineWidth', 1.5);
    plot(log1.t, log1.w(:,3), '-', 'LineWidth', 1.5);
    xlabel('时间 (s)'); ylabel('份额'); title('NN 在线功率份额 w(t)'); grid on; legend('w1','w2','w3');

    %% ---------- 7) 返回指标 ----------
    metrics = struct( ...
        'row', n, ...
        't_baseline', t0, 't_spinn_online', t1, 'timesaved', timesaved, ...
        'v_baseline', v0, 'v_spinn_online', v1, 'speed_reduction', speed_red, ...
        'w_baseline', w_baseline, 'w0_pred', w0, ...
        'joint_cfg', joint_cfg, 'cfg_common', cfg_common);
end

% ---- 工具 ----
function w = project_to_simplex_row(w)
    w = max(w(:).', 0); s = sum(w);
    if s <= 0, w = ones(size(w))/numel(w); else, w = w/s; end
end
