function outPath = spinn_hnn_make_dataset(Ntraj, outPath, varargin)
% spinn_hnn_make_dataset
% 生成 H_θ（哈密顿网络）训练数据集：{Q,DQ,P,DPDT,TAU,Rdiag,Qn,Pn,dt,L,g}
%
% I/O
%   Ntraj   : 需要生成的“轨迹条数”（建议 32~256）
%   outPath : 输出 .mat 路径（默认 'spinn_hnn_ds.mat'）
% 可选名值对：
%   'seed'      : 随机种子（可复现）
%   'cfg'       : 结构体，覆盖仿真口径（默认 dt=0.002, t_final=5.0, radius=0.2）
%   'fix_dq0'   : 1x3，固定初始角速度（默认 [0 0 0]，更稳）
%   'Kp','Ki','Kd' : PID 参数（默认 [60 60 60], [0.20 0.20 0.20], [0.10 0.10 0.10]）
%   'w'         : 1x3，老师核用的功率份额（默认均分）
%
% 依赖：spinn_RandomNumberGeneration / spinn_MechanicAlarm / computeDynamics
% 说明：本函数只“造数据”，后续训练 H_θ 与“辛损失”在独立脚本中进行。

    % ---------- 0) 解析参数 ----------
    if nargin < 1 || isempty(Ntraj), Ntraj = 64; end
    if nargin < 2 || isempty(outPath), outPath = 'spinn_hnn_ds.mat'; end
    p = inputParser; p.KeepUnmatched=true;
    addParameter(p,'seed',[],@(x) isempty(x) || isscalar(x));
    addParameter(p,'cfg', struct('dt',0.002,'t_final',5.0,'radius',0.2));
    addParameter(p,'fix_dq0',[0 0 0],@(x) isnumeric(x) && numel(x)==3);
    addParameter(p,'Kp',[60 60 60]);
    addParameter(p,'Ki',[0.20 0.20 0.20]);
    addParameter(p,'Kd',[0.10 0.10 0.10]);
    addParameter(p,'w',[1 1 1]/3);
    parse(p,varargin{:});
    opt = p.Results;

    % ---------- 1) 采样 25 维参数（与工程 schema 一致） ----------
    PM25 = spinn_RandomNumberGeneration(Ntraj, 'fix_dq0', opt.fix_dq0, 'seed', opt.seed);  % 25 列齐全

    % ---------- 2) 常量与存储 ----------
    g = 9.81; L = [0.24, 0.214, 0.324];      % 与动力学核一致
    Q=[]; DQ=[]; P=[]; DPDT=[]; TAU=[]; Rdiag=[]; Qn=[]; Pn=[];
    bad_total = 0;

    % ---------- 3) 逐条轨迹：老师核仿真 → 组装 (q,dq,p,dp/dt,τ) ----------
    for i = 1:size(PM25,1)
        row = PM25(i,:);
        m       = row(1:3);
        dq0     = row(4:6);
        dampNom = [row(7) row(9) row(11)];       % 名义阻尼（三轴）
        initDeg = row(13:15);
        tgtDeg  = row(16:18);
        Pmax    = row(22);

        % —— 28 维参数：与 spinn_MechanicAlarm 完全对齐（PID + w）——
        params28 = [ m, dq0, ...
                     dampNom, tgtDeg, initDeg, Pmax, ...
                     opt.Kp(1), opt.Ki(1), opt.Kd(1), ...
                     opt.Kp(2), opt.Ki(2), opt.Kd(2), ...
                     opt.Kp(3), opt.Ki(3), opt.Kd(3), ...
                     normalize_w(opt.w) ];

        % —— 仿真（半隐式/辛 Euler；返回完整轨迹与 τ/功率诊断）——
        try
            [~, info] = spinn_MechanicAlarm(params28, opt.cfg);
        catch ME
            warning('[%d/%d] spinn_MechanicAlarm 失败：%s', i, size(PM25,1), ME.message);
            bad_total = bad_total + 1; 
            continue;
        end

        % 时间与状态
        t  = info.t(:);    if numel(t) < 3, continue; end
        dt = mean(diff(t));   % 保持与 cfg 一致
        qh = info.q_history;      % k x 3 (rad)
        dqh= info.dq_history;     % k x 3 (rad/s)
        tau= info.tau_history;    % k x 3 (N·m)

        % —— 动量 p = M(q)dq（采用你工程的 computeDynamics）——
        K = size(qh,1);
        ptraj = zeros(K,3);
        for k = 1:K
            [Mq, ~, ~] = computeDynamics(m(1),m(2),m(3), L(1),L(2),L(3), g, qh(k,:).', dqh(k,:).');
            ptraj(k,:) = (Mq * dqh(k,:).').';
        end

        % —— 中心差分近似 dp/dt（端点复制）——
        dpdt = zeros(K,3);
        dpdt(2:K-1,:) = (ptraj(3:end,:) - ptraj(1:end-2,:)) / (2*dt);
        dpdt(1,:)     = dpdt(2,:); 
        dpdt(K,:)     = dpdt(K-1,:);

        % —— 组装 (k, k+1) 配对样本（供能量差分项使用）——
        Q   = [Q;   qh(1:K-1,:)];           % 当前 q_k
        DQ  = [DQ;  dqh(1:K-1,:)];          % 当前 dq_k
        P   = [P;   ptraj(1:K-1,:)];        % 当前 p_k
        DPDT= [DPDT;dpdt(1:K-1,:)];         % 当前 \dot p_k
        TAU = [TAU; tau(1:K-1,:)];          % 当前 τ_k
        Rdiag=[Rdiag; repmat(dampNom, K-1, 1)];   % 阻尼对角（名义）
        Qn  = [Qn;  qh(2:K,:)];             % 下一时刻 q_{k+1}
        Pn  = [Pn;  ptraj(2:K,:)];          % 下一时刻 p_{k+1}
    end

    % ---------- 4) 清洗与落盘 ----------
    ALL = [Q DQ P DPDT TAU Rdiag Qn Pn];
    good = all(isfinite(ALL),2);
    bad = nnz(~good); 
    if bad>0
        warning('清理掉 %d 条含 NaN/Inf 的样本。', bad);
    end
    Q=Q(good,:); DQ=DQ(good,:); P=P(good,:); DPDT=DPDT(good,:); 
    TAU=TAU(good,:); Rdiag=Rdiag(good,:); Qn=Qn(good,:); Pn=Pn(good,:);

    save(outPath,'Q','DQ','P','DPDT','TAU','Rdiag','Qn','Pn','dt','L','g','-v7.3');
    fprintf('[spinn_hnn_make_dataset] 完成：Ntraj=%d → 样本数=%d，保存到：%s\n', Ntraj, size(Q,1), outPath);
    if bad_total>0
        fprintf('注意：有 %d 条轨迹在仿真阶段失败/跳过。\n', bad_total);
    end
end

% === 工具：份额归一化 ===
function w = normalize_w(w)
    w = max(w(:).',0);
    s = sum(w); if s<=0, w = ones(size(w))/numel(w); else, w = w/s; end
end
