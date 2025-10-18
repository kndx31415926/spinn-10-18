function opts = spinn_defaults(opts)
% 全局默认的 SPINN 仿真选项；任何模块都可调用
% 字段含义与我们在 spinn_MechanicAlarm/optimizemechanicalarm 中一致
    if nargin < 1 || isempty(opts), opts = struct(); end
    def = struct( ...
        'dt',        0.002, ...   % 步长
        't_final',   10,   ...   % 仿真截止时间
        'radius',    0.01,   ...   % 命中判定半径
        'lambda_v',  0.25,  ...   % 终端速度惩罚系数（老师目标）
        'ou_tau',    0.30,  ...   % 阻尼 OU 抖动时间常数
        'k_dir_max', 25     ...   % 几何最速力矩的放大上限
    );
    f = fieldnames(def);
    for i = 1:numel(f)
        if ~isfield(opts, f{i}) || isempty(opts.(f{i}))
            opts.(f{i}) = def.(f{i});
        end
    end
end