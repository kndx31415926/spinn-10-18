function PM25 = spinn_RandomNumberGeneration(n, varargin)
% SPINN 参数采样器（25维）
% 输出矩阵每行 25 列，与数据生成/训练脚本严格对齐：
% [1..3]   m1 m2 m3
% [4..6]   dq0_1 dq0_2 dq0_3
% [7..12]  damp1 zeta1 damp2 zeta2 damp3 zeta3
% [13..15] init_deg1 init_deg2 init_deg3
% [16..18] tgt_deg1  tgt_deg2  tgt_deg3
% [19..21] dtheta1 dtheta2 dtheta3   ← 程序计算: tgt - init
% [22]     Pmax
% [23..25] Prated1 Prated2 Prated3   (需满足 Prated_i ≤ Pmax)
%
% 用法：
%   PM25 = spinn_RandomNumberGeneration(200);               % 默认设置
%   PM25 = spinn_RandomNumberGeneration(200,'use_lhs',false,'batch',512);
%   PM25 = spinn_RandomNumberGeneration(200,'fix_pid',[50 0.2 0.1]); % 仅用于对齐生成固定 PID 的上游脚本
%   PM25 = spinn_RandomNumberGeneration(200,'fix_init',[0 0 0],'fix_dq0',[0 0 0]);
%
% 说明：
% - 维度与列含义与 `spinn_DatasetGeneration`、`spinn_train_model` 完全一致，直接串联可用。
% - 采样器不对 19..21 列采样，而是根据 16..18 与 13..15 自动填充 ∆θ。 
% - 强约束：目标角严格大于初始角；各轴 Prated ≤ Pmax。 
%
% (c) SPINN stack - sampler

    % -------- 0) 解析 n 与可选项 --------
    if nargin < 1 || isempty(n), n = 200; end
    if ~isscalar(n) || ~isfinite(n) || n<1, n = 200; end
    n = round(n);

    % 默认选项
    opts = struct( ...
        'batch',     max(256, ceil(n/5)), ...   % 每批候选数量
        'max_tries', 100,                ...    % 最多补批次数
        'use_lhs',   true,               ...    % 有 lhsdesign 则用
        'fix_pid',   [],                 ...    % 1x3 or 1x9，指定 PID（仅为与上游脚本保持口径；不进入 25 维）
        'fix_init',  [],                 ...    % 1x3 固定初始角(°)，留空则按范围采样/默认
        'fix_dq0',   [],                 ...    % 1x3 固定初始角速度(rad/s)，留空则按范围
        'seed',      []);                      % 设定随机种子（可复现）
    opts = parse_opts(opts, varargin{:});

    if ~isempty(opts.seed)
        rng(double(opts.seed));
    else
        rng('shuffle');
    end

    % -------- 1) 取值范围（与现工程一致的稳健范围）--------
    % 注：dq0 默认采样为 0（与对比脚本口径一致）；可通过 fix_dq0 覆盖
    lb = [ 0.16  0.18  1.00,  ... m1..m3
           0.00  0.00  0.00,  ... dq0_1..3
           3.00  0.00, 3.00  0.00, 3.00  0.00, ... damp_i, zeta_i
           49.00  49.00  49.00,  ... init_deg1..3（默认0°）
           55.0  55.0  55.0,  ... tgt_deg1..3
           0 0 0,            ... dtheta （占位）
           200,              ... Pmax
           10    10    10 ]; ... Prated1..3
    ub = [ 0.20  0.25  2.00, ...
           4.00  4.00  4.00, ... ← 默认将 dq0 固定为 0
           4.00  0.60, 4.00  0.60, 4.00  0.60, ...
           50.00  50.00  50.00, ...
           60.0  60.0  90.0, ...
           180  180  180, ...
           220, ...
           120   120   120 ];
    D = numel(lb);  assert(D==25, '内部维度应为 25。');

    % 采样列（去掉 dtheta 的 19..21）
    idx_dtheta = 19:21;
    idx_samp   = setdiff(1:D, idx_dtheta);
    Ds = numel(idx_samp);

    % 若用户指定 fix_init / fix_dq0，则同步修改相应上下限
    if ~isempty(opts.fix_init)
        fi = as_row(opts.fix_init, 3);
        lb(13:15) = fi; ub(13:15) = fi;
    end
    if ~isempty(opts.fix_dq0)
        fd = as_row(opts.fix_dq0, 3);
        lb(4:6) = fd;  ub(4:6) = fd;
    end

    % -------- 2) 分批补齐，不足则多次尝试 --------
    PM25  = zeros(0, D);
    tries = 0;

    while size(PM25,1) < n && tries < opts.max_tries
        tries = tries + 1;

        m = max(opts.batch, n - size(PM25,1));    % 本批候选

        % ---- 生成 U in [0,1]^(Ds) ----
        U = [];
        if opts.use_lhs && exist('lhsdesign','file')==2
            try
                U = lhsdesign(m, Ds, 'smooth','off','criterion','none','iterations',0);
            catch
                U = lhsdesign(m, Ds);  % 回退（不同版本接口不同）
            end
        end
        if isempty(U), U = rand(m, Ds); end

        % ---- 映射到 [lb, ub] ----
        Xs = U .* (ub(idx_samp) - lb(idx_samp)) + lb(idx_samp);

        % ---- 组装回 25 维，并计算 Δθ=tgt-init（不采样） ----
        Xfull = zeros(m, D);
        Xfull(:, idx_samp) = Xs;

        % dtheta = tgt - init（单位：度）
        Xfull(:, idx_dtheta) = Xfull(:,16:18) - Xfull(:,13:15);

        % ---- 约束过滤 ----
        valid = true(m,1);

        % 目标角 > 初始角（逐轴）
        valid = valid & (Xfull(:,16) > Xfull(:,13)) & (Xfull(:,17) > Xfull(:,14)) & (Xfull(:,18) > Xfull(:,15));

        % 各轴额定功率 ≤ Pmax
        valid = valid & all( Xfull(:,23:25) <= Xfull(:,22) + 1e-12, 2);

        % 基本范围检查（浮点容差）
        valid = valid & all( Xfull(:,1:D) >= (lb - 1e-12), 2) & all( Xfull(:,1:D) <= (ub + 1e-12), 2);

        Xfull = Xfull(valid, :);

        % 追加本批的前 need 行
        need = n - size(PM25,1);
        PM25 = [PM25; Xfull(1:min(need, size(Xfull,1)), :)]; %#ok<AGROW>
    end

    if size(PM25,1) < n
        warning('仅采样到 %d/%d 行；可提高 max_tries 或放宽范围。', size(PM25,1), n);
    end
end

% ======================= 工具函数 =======================

function opts = parse_opts(opts, varargin)
    if isempty(varargin), return; end
    % 支持：结构体或名值对
    if isstruct(varargin{1})
        s = varargin{1}; f = fieldnames(s);
        for i=1:numel(f), opts.(f{i}) = s.(f{i}); end
        return;
    end
    if mod(numel(varargin),2)~=0
        error('可选参数必须为名-值对或单个结构体。');
    end
    for k=1:2:numel(varargin)
        name = lower(string(varargin{k}));
        val  = varargin{k+1};
        switch name
            case "batch",      opts.batch = double(val);
            case {"max_tries","maxtries"}, opts.max_tries = double(val);
            case {"use_lhs","uselhs"},     opts.use_lhs   = logical(val);
            case "fix_pid",    opts.fix_pid = val;        % 仅保留，避免上游脚本断言；本采样器不使用 PID
            case "fix_init",   opts.fix_init = val;
            case "fix_dq0",    opts.fix_dq0  = val;
            case "seed",       opts.seed = val;
            otherwise, warning('未知选项 %s 已忽略。', name);
        end
    end
end

function r = as_row(x, n)
    x = double(x(:).');
    if numel(x)==1, r = repmat(x, 1, n);
    elseif numel(x)==n, r = x;
    else, error('期望长度为 %d 的向量或标量。', n);
    end
end
