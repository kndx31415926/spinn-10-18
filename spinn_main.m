function spinn_main(n, num_trials, out_path)
% SPINN 数据生成主程序（采样 → 逐行优化落盘 → 清洗）
% 输出：C:\Users\kndx9\Desktop\SpinnMechanicalArmParams.mat（可自定义）
% 布局：1..25=输入，26..28=最优权重，29=最短到达时间

    % ---- 参数与默认值 ----
    if nargin < 1 || isempty(n), n = 3000; end
    if nargin < 2 || isempty(num_trials), num_trials = 10; end
    if nargin < 3 || isempty(out_path)
        out_path = fullfile('C:','Users','kndx9','Desktop','SpinnMechanicalArmParams.mat');
    end
    n = max(1, round(double(n)));
    num_trials = max(1, round(double(num_trials)));
    fprintf('[spinn_main] 目标样本数: %d, 每行重启: %d\n', n, num_trials);
    fprintf('[spinn_main] 输出文件: %s\n', out_path);

    % ---- 0) 预建空文件（无则建） ----
    ensure_outfile(out_path);

    % ---- 1) 采样 ----
    sampler_opts = struct('batch', max(256, ceil(n/5)), 'max_tries', 100, 'use_lhs', true);
    try
        PM25 = spinn_RandomNumberGeneration(n, sampler_opts);
    catch ME
        warning('[spinn_main] 采样器带参失败：%s → 回退单参。', ME.message);
        PM25 = spinn_RandomNumberGeneration(n);
    end
    if isempty(PM25) || size(PM25,2) ~= 25
        error('[spinn_main] 采样失败或维度异常（期望 25 列）。');
    end
    fprintf('[spinn_main] 采样完成：%d 行 x %d 列。\n', size(PM25,1), size(PM25,2));

    % ---- 2) 逐行优化 + 追加落盘 ----
    m = size(PM25,1);
    n_failed = 0;
    for i = 1:m
        try
            spinn_DatasetGeneration(PM25(i,:), num_trials, out_path);
            if mod(i,10)==0 || i==m
                fprintf('[spinn_main] 已成功追加至第 %d/%d 行。\n', i, m);
            end
        catch ME
            n_failed = n_failed + 1;
            warning('[spinn_main] 第 %d/%d 行失败：%s', i, m, ME.message);
        end
        if mod(i, 20) == 0
            close all; drawnow;
        end
    end
    if n_failed>0
        warning('[spinn_main] 共 %d 行失败（请上滚查看第一条失败原因）。', n_failed);
    end

    % ---- 3) 清洗（去 NaN/0 行；仅对本 SPINN 文件）----
    try
        spinn_cleandata(out_path);
    catch ME
        warning('[spinn_main] spinn_cleandata 失败：%s', ME.message);
    end

    % ---- 4) 汇总信息 ----
    try
        if isfile(out_path)
            S = load(out_path);
            if isfield(S,'params_matrix')
                fprintf('[spinn_main] 产出: %s; 大小: %dx%d\n', out_path, size(S.params_matrix,1), size(S.params_matrix,2));
            else
                warning('[spinn_main] %s 中未找到变量 params_matrix。', out_path);
            end
        else
            warning('[spinn_main] 未找到输出 MAT 文件: %s', out_path);
        end
    catch ME
        warning('[spinn_main] 汇总信息读取失败：%s', ME.message);
    end
end

% --- 创建空输出文件（带 params_matrix 变量），目录不存在则 mkdir ---
function ensure_outfile(fp)
    folder = fileparts(fp);
    if ~exist(folder,'dir'), mkdir(folder); end
    if ~isfile(fp)
        params_matrix = []; %#ok<NASGU>
        save(fp, 'params_matrix', '-v7.3');
        fprintf('[spinn_main] 已创建空文件：%s\n', fp);
    end
end
