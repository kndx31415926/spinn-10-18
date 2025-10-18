function [w_opt, t_best] = spinn_optimizemechanicalarm(fixed25, num_trials, Prated, Pmax, varargin)
% SPINN 版权重优化：在功率双顶帽约束下最小化“到达时间”
%
% 输入
%   fixed25   : 1x25，物理核的前25维（不含 w）
%   num_trials: 重启次数（默认 10）
%   Prated    : 1x3，各轴额定功率上限（W）
%   Pmax      : 标量，总功率上限（W）
% 名值可选
%   'cfg'       : 结构体，传给 spinn_MechanicAlarm（默认 dt=0.002, t_final=5.0, radius=0.2）
%   'objective' : 'time'（默认）或 'joint'（时间 + λ·末端命中速度）
%   'lambda_v'  : joint 目标的速度权重 λ（默认 0.0）
%
% 输出
%   w_opt  : 1x3，最优功率份额（满足 0≤w≤Prated/Pmax，∑w=1）
%   t_best : 标量，最短到达时间（若不可达，返回 t_final+1 的有限惩罚值）

    % ---------- 1) 解析参数 ----------
    if nargin < 2 || isempty(num_trials), num_trials = 10; end
    if nargin < 3 || isempty(Prated),    Prated = [Inf,Inf,Inf]; end
    if nargin < 4 || isempty(Pmax),      Pmax   =  Inf; end
    assert(isvector(fixed25) && numel(fixed25)==25, 'fixed25 必须是 1x25。');

    opts = struct('cfg',[], 'objective','time', 'lambda_v', 0.0);
    if ~isempty(varargin)
        if mod(numel(varargin),2)~=0, error('名值对必须成对出现。'); end
        for k=1:2:numel(varargin)
            key = lower(string(varargin{k}));
            val = varargin{k+1};
            switch key
                case "cfg",        opts.cfg = val;
                case "objective",  opts.objective = char(val);
                case "lambdav",    opts.lambda_v  = double(val);
                otherwise, warning('未知选项 %s 已忽略。', key);
            end
        end
    end
    if isempty(opts.cfg), opts.cfg = local_defaults(); end
    % 统一目标
    use_joint = strcmpi(opts.objective,'joint');
    lambda_v  = max(0.0, double(opts.lambda_v));

    % ---------- 2) 选择仿真核（首选 SPINN 核） ----------
    sim_fun = pick_simulator(fixed25, opts.cfg);   % 返回 handle: [loss, t_raw] = sim_fun(w)

    % ---------- 3) 上界（由额定功率） ----------
    cap = Prated(:).' / max(Pmax, eps);
    cap(~isfinite(cap)) = 1;
    cap = max(0, min(1, cap));

    % ---------- 4) 多重重启 + 投影梯度 ----------
    t_best = inf; 
    w_opt  = [1,0,0];     % 占位
    seeds  = [ ones(1,3)/3; cap/sum(cap); rand(1,3) ];  % 均分 / 额定比例 / 随机
    n_seed = size(seeds,1);

    for tr = 1:max(num_trials, n_seed)
        if tr <= n_seed, w = seeds(tr,:); else, w = rand(1,3); end
        w = project_capped_simplex(w, cap);

        lr = 0.05; maxit = 300; tol = 1e-5; last = inf;
        for it = 1:maxit
            f = eval_loss(sim_fun, w, use_joint, lambda_v);       % 标量损失（已含惩罚）
            g = fd_grad(@(z) eval_loss(sim_fun, project_capped_simplex(z,cap), use_joint, lambda_v), w);

            % 一步下降 + 带上界投影
            w_new = project_capped_simplex(w - lr*g, cap);

            % 简易回溯线搜
            f_new = eval_loss(sim_fun, w_new, use_joint, lambda_v);
            bt=0;
            while f_new > f && bt < 6
                lr = lr * 0.5;
                w_new = project_capped_simplex(w - lr*g, cap);
                f_new = eval_loss(sim_fun, w_new, use_joint, lambda_v);
                bt = bt + 1;
            end
            w = w_new; f = f_new;

            if abs(last - f) < tol, break; end
            last = f;
        end

        % 记录最好（注意：f 已是“惩罚后”的有限标量）
        if f < t_best
            t_best = f;
            w_opt  = w;
        end
    end

    % ---------- 5) 返回 ----------
    % t_best 已为有限（可达=真实时间；不可达= t_final+1 惩罚）
end

% ====== 内部：默认 cfg（与 compare 链口径一致） ======
function cfg = local_defaults()
    cfg = struct('dt',0.002, 't_final',5.0, 'radius',0.2, 'omega_eps',1e-3);
end

% ====== 内部：选择仿真核，返回“带惩罚/联合目标”的评估器 ======
function sim_fun = pick_simulator(fixed25, cfg)
    if exist('spinn_MechanicAlarm','file') == 2
        % 用 SPINN 核，显式传 cfg（关键修复点）
        sim_fun = @(w) sim_spinn([fixed25, w], cfg);
    elseif exist('MechanicAlarm','file') == 2
        % 兼容旧核（无 cfg），只拿时间
        sim_fun = @(w) sim_old([fixed25, w]);
    else
        % 兜底：mechanical_arm2（只在你的老工程中用到）
        assert(exist('mechanical_arm2','file')==2, '未找到仿真核。');
        sim_fun = @(w) sim_fallback([fixed25, w]);
    end
end

% --- 用 SPINN 核：返回 [t_raw, v_end]；失败返回 NaN ---
function [t_raw, v_end] = sim_spinn(full28, cfg)
    try
        % 尽量取 info.end_speed；若旧版仅返回时间，则 v_end=NaN
        try
            [t_raw, info] = spinn_MechanicAlarm(full28, cfg);
            if isstruct(info) && isfield(info,'end_speed'), v_end = info.end_speed; else, v_end = NaN; end
        catch
            t_raw = spinn_MechanicAlarm(full28, cfg); v_end = NaN;
        end
    catch
        t_raw = NaN; v_end = NaN;
    end
end

% --- 旧核（无 cfg）：只返回时间 ---
function [t_raw, v_end] = sim_old(full28)
    try
        t_raw = MechanicAlarm(full28); 
    catch
        t_raw = NaN;
    end
    v_end = NaN;
end

% --- 兜底 ---
function [t_raw, v_end] = sim_fallback(full28)
    try
        [~,~,~,~,t_raw] = mechanical_arm2(full28);
    catch
        t_raw = NaN;
    end
    v_end = NaN;
end

% ====== 内部：将“原始时间/末速”映射为优化损失（含惩罚） ======
function loss = eval_loss(sim_fun, w, use_joint, lambda_v)
    [t_raw, v_end] = sim_fun(w);

    % 惩罚映射：非有限或非正 → t_penalty
    [cfg_dt, cfg_T, cfg_R] = deal(0.002, 5.0, 0.2); %#ok<NASGU> % 仅用于可读；t_penalty 与 local_defaults 一致
    t_penalty = 60;
    if ~isfinite(t_raw) || t_raw <= 0
        loss = t_penalty;
        return;
    end

    if ~use_joint
        loss = t_raw;
    else
        v = v_end; if ~isfinite(v), v = 0.0; end
        loss = t_raw + lambda_v * max(0,v);
    end
end

% ====== 内部：中心差分数值梯度 ======
function g = fd_grad(fun, w)
    h = 1e-2; g = zeros(size(w));
    for i=1:numel(w)
        e = zeros(size(w)); e(i)=1;
        g(i) = (fun(w + h*e) - fun(w - h*e)) / (2*h);
    end
end

% ====== 内部：带上界的单纯形投影（∑=1, 0≤w≤cap），稳定二分 ======
function w = project_capped_simplex(w, cap)
    w   = max(w(:).', 0);
    cap = max(cap(:).', 0); cap(~isfinite(cap)) = 1;

    if sum(cap) < 1 - 1e-12
        % 可行域为空：退化为 cap 的相对比例
        if sum(cap) <= 0, w = ones(1,numel(w))/numel(w); else, w = cap/sum(cap); end
        return;
    end

    lo = min(w - cap); hi = max(w);
    for it=1:60
        mu = 0.5*(lo+hi);
        v  = min(max(w - mu, 0), cap);
        s  = sum(v);
        if abs(s - 1) < 1e-12, w = v; return; end
        if s > 1, lo = mu; else, hi = mu; end
    end
    w = min(max(w - mu, 0), cap);  % 容错
end
