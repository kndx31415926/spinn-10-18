function model = spinn_hnn_train(dsPath, outPath, varargin)
% spinn_hnn_train
% 训练 H_θ(q,p)（标量网络），以“动力学一致 + 能量流（辛）损失”优化：
%   L = Lq + λp*Lp + λE*LE
%   其中：
%     Lq = || ∂H/∂p - dq ||^2
%     Lp = || dpdt + ∂H/∂q - τ + R * ∂H/∂p ||^2
%     LE = || (H(z_{k+1}) - H(z_k))/dt - <∂H/∂p, τ> + <∂H/∂p, R∂H/∂p> ||^2
%
% 输入
%   dsPath  : 由 spinn_hnn_make_dataset 生成的数据集 .mat（含 Q,DQ,P,DPDT,TAU,Rdiag,Qn,Pn,dt）
%   outPath : 输出模型文件（默认 'spinn_hnn_model.mat'）
% 可选名值对
%   'epochs'   : 轮数（默认 50）
%   'mb'       : batch size（默认 1024）
%   'lr'       : Adam 学习率（默认 1e-3）
%   'lambda_p' : Lp 权重（默认 1.0）
%   'lambda_E' : LE 权重（默认 0.1）
%   'seed'     : 随机种子（默认 [] 不设）
%   'useGPU'   : 是否用 GPU（默认 canUseGPU()）
%
% 输出
%   model : 结构体，含 dlnet、muX、sigmaX、dt、loss 曲线等。并保存到 outPath。

    % ---------- 0) 参数 ----------
    if nargin<1 || isempty(dsPath), dsPath = 'spinn_hnn_ds.mat'; end
    if nargin<2 || isempty(outPath), outPath = 'spinn_hnn_model.mat'; end
    opts = struct('epochs',50,'mb',1024,'lr',1e-3,'lambda_p',1.0,'lambda_E',0.1,'seed',[],'useGPU',[]);
    opts = parseOpts(opts, varargin{:});
    if isempty(opts.useGPU), opts.useGPU = canUseGPU(); end
    if ~isempty(opts.seed), rng(double(opts.seed)); end

    % ---------- 1) 载入数据 ----------
    S = load(dsPath);
    need = {'Q','DQ','P','DPDT','TAU','Rdiag','Qn','Pn','dt'};
    for i=1:numel(need), assert(isfield(S,need{i}), '数据集中缺少字段: %s', need{i}); end
    Q=S.Q; DQ=S.DQ; P=S.P; DP=S.DPDT; TAU=S.TAU; Rdiag=S.Rdiag; Qn=S.Qn; Pn=S.Pn; dt=S.dt;

    % 基本清洗
    good = all(isfinite([Q DQ P DP TAU Rdiag Qn Pn]),2);
    Q=Q(good,:); DQ=DQ(good,:); P=P(good,:); DP=DP(good,:);
    TAU=TAU(good,:); Rdiag=Rdiag(good,:); Qn=Qn(good,:); Pn=Pn(good,:);

    % ---------- 2) 归一化（仅对输入 [q,p] 做 z-score） ----------
    X = [Q P];
    muX = mean(X,1);  sigmaX = std(X,0,1)+1e-6;      % 6 维
    muXv = dlarray(muX(:), 'CB');                    % 6x1
    sigXv= dlarray(sigmaX(:), 'CB');                 % 6x1
    if opts.useGPU, muXv = gpuArray(muXv); sigXv = gpuArray(sigXv); end
    normZ = @(q,p) (([q; p] - muXv) ./ sigXv);       % 输入标准化（可微）

    % ---------- 3) 划分训练/验证 ----------
    N = size(Q,1);
    idx = randperm(N);
    ntr = round(0.85*N);
    id_tr = idx(1:ntr); id_va = idx(ntr+1:end);

    % ---------- 4) 定义网络 H_θ：6→128→128→64→1 ----------
    hasTanh = (exist('tanhLayer','file') == 2);
    if hasTanh
        act1 = tanhLayer('Name','t1'); act2 = tanhLayer('Name','t2'); act3 = tanhLayer('Name','t3');
    else
        act1 = reluLayer('Name','t1_relu'); act2 = reluLayer('Name','t2_relu'); act3 = reluLayer('Name','t3_relu');
    end
    layers = [
        featureInputLayer(6,'Normalization','none','Name','in')
        fullyConnectedLayer(128,'Name','fc1')
        act1
        fullyConnectedLayer(128,'Name','fc2')
        act2
        fullyConnectedLayer(64,'Name','fc3')
        act3
        fullyConnectedLayer(1,'Name','out')
    ];
    dlnet = dlnetwork(layers);

    % ---------- 5) 训练循环（Adam） ----------
    trailingAvg = []; trailingAvgSq = [];
    trainLoss = zeros(opts.epochs,1); valLoss = zeros(opts.epochs,1);
    mb = opts.mb; useGPU = opts.useGPU;
    globalStep = int32(0);   % 关键：迭代步（供部分 adamupdate 版本要求“整数”）

    for ep = 1:opts.epochs
        id = id_tr(randperm(numel(id_tr)));
        loss_acc = 0; nstep = 0;

        for s = 1:mb:numel(id)
            mbId = id(s : min(s+mb-1, numel(id)));
            [q,p,dq,dp,tau,rD,qn,pn] = getBatch(mbId, Q,P,DQ,DP,TAU,Rdiag,Qn,Pn, useGPU);

            [L,grad,stats] = dlfeval(@modelLoss, dlnet, q,p,dq,dp,tau,rD, qn,pn, dt, normZ, opts);

            % —— 兼容多签名 adamupdate；若均失败则手写 Adam 更新 ——
            globalStep = globalStep + 1;
            [dlnet, trailingAvg, trailingAvgSq] = local_adamupdate(dlnet, grad, trailingAvg, trailingAvgSq, opts.lr, globalStep);

            loss_acc = loss_acc + double(L);
            nstep = nstep + 1;

            if mod(nstep,10)==1 || s==1
                fprintf('ep %02d | it %4d | L=%.3e (Lq=%.2e Lp=%.2e LE=%.2e)\n', ...
                        ep, nstep, double(L), double(stats.Lq), double(stats.Lp), double(stats.LE));
            end
        end

        trainLoss(ep) = loss_acc / max(nstep,1);

        % —— 验证集（单 batch 估计）——
        if ~isempty(id_va)
            vaId = id_va(1:min(numel(id_va), max(32, mb)));
            [q,p,dq,dp,tau,rD,qn,pn] = getBatch(vaId, Q,P,DQ,DP,TAU,Rdiag,Qn,Pn, useGPU);
            [Lv,~,statv] = dlfeval(@modelLoss, dlnet, q,p,dq,dp,tau,rD, qn,pn, dt, normZ, opts);
            valLoss(ep) = double(Lv);
            fprintf('>> ep %02d done | Train=%.3e | Val=%.3e | (Lq=%.2e Lp=%.2e LE=%.2e)\n', ...
                    ep, trainLoss(ep), valLoss(ep), double(statv.Lq), double(statv.Lp), double(statv.LE));
        else
            valLoss(ep) = NaN;
            fprintf('>> ep %02d done | Train=%.3e | Val=N/A (no val set)\n', ep, trainLoss(ep));
        end
    end

    % ---------- 6) 保存 ----------
    model = struct('dlnet',dlnet,'muX',muX,'sigmaX',sigmaX,'dt',dt, ...
                   'trainLoss',trainLoss,'valLoss',valLoss,'arch','HNet(6-128-128-64-1,act=tanh/relu)');
    save(outPath, '-struct', 'model');
    fprintf('[spinn_hnn_train] 已保存模型到：%s\n', outPath);
end

% ================= 内部函数 =================

function [L,grad,stats] = modelLoss(net, q,p,dq,dp,tau,rD, qn,pn, dt, normZ, opts)
    % 归一化后的输入
    z  = normZ(q, p);        % 6xB (CB)
    zn = normZ(qn, pn);      % 6xB

    % 前向
    H  = forward(net, z);    % 1xB (CB)
    Hn = forward(net, zn);   % 1xB

    % 对输入求导（需高阶导用于参数反传）
    Hsum = sum(H,'all');
    [dHdq, dHdp] = dlgradient(Hsum, q, p, 'EnableHigherDerivatives', true);  % 3xB, 3xB

    % 三项损失
    Lq = mean(sum((dHdp - dq).^2, 1), 'all');
    Lp = mean(sum((dp + dHdq - tau + rD.*dHdp).^2, 1), 'all');

    dHdt = (Hn - H) ./ dt;             % 1xB
    power = sum(dHdp .* tau, 1);       % 1xB  ≈ ωᵀτ
    diss  = sum(dHdp .* (rD.*dHdp), 1);% 1xB  ≈ ωᵀRω
    LE = mean((dHdt - power + diss).^2, 'all');

    L = Lq + opts.lambda_p * Lp + opts.lambda_E * LE;

    % 对网络参数求梯度（通过 dH/dq, dH/dp → 需高阶导）
    grad = dlgradient(L, net.Learnables, 'EnableHigherDerivatives', true);

    stats = struct('Lq',Lq,'Lp',Lp,'LE',LE);
end

function [q,p,dq,dp,tau,rD,qn,pn] = getBatch(ids, Q,P,DQ,DP,TAU,Rdiag,Qn,Pn, useGPU)
    % 先放 GPU，再封装 dlarray（保证在 GPU 上跟踪梯度）
    q  = Q(ids,:).';   p  = P(ids,:).';
    dq = DQ(ids,:).';  dp = DP(ids,:).';
    tau= TAU(ids,:).'; rD = Rdiag(ids,:).';
    qn = Qn(ids,:).';  pn = Pn(ids,:).';

    if useGPU
        q=gpuArray(q); p=gpuArray(p); dq=gpuArray(dq); dp=gpuArray(dp);
        tau=gpuArray(tau); rD=gpuArray(rD); qn=gpuArray(qn); pn=gpuArray(pn);
    end

    q  = dlarray(q,'CB');   p  = dlarray(p,'CB');
    dq = dlarray(dq,'CB');  dp = dlarray(dp,'CB');
    tau= dlarray(tau,'CB'); rD = dlarray(rD,'CB');
    qn = dlarray(qn,'CB');  pn = dlarray(pn,'CB');
end

function o = parseOpts(o, varargin)
    if isempty(varargin), return; end
    assert(mod(numel(varargin),2)==0, '名值对必须成对传入。');
    for k=1:2:numel(varargin)
        o.(char(varargin{k})) = varargin{k+1};
    end
end

% ===== 多版本 adamupdate 适配：依次尝试多种签名；若均失败则手写 Adam =====
function [net, tAvg, tAvgSq] = local_adamupdate(net, grad, tAvg, tAvgSq, lr, iterInt)
    % 1) (net,grad,avg,avgSq,iteration) —— 第 5 参为整数
    try
        [net, tAvg, tAvgSq] = adamupdate(net, grad, tAvg, tAvgSq, int32(iterInt));
        return;
    catch, end
    % 2) (net,grad,avg,avgSq,iteration,learnRate)
    try
        [net, tAvg, tAvgSq] = adamupdate(net, grad, tAvg, tAvgSq, int32(iterInt), lr);
        return;
    catch, end
    % 3) (net,grad,avg,avgSq,learnRate)
    try
        [net, tAvg, tAvgSq] = adamupdate(net, grad, tAvg, tAvgSq, lr);
        return;
    catch, end
    % 4) (net,grad,avg,avgSq,learnRate,beta1,beta2,epsilon,iteration) —— 旧版
    try
        [net, tAvg, tAvgSq] = adamupdate(net, grad, tAvg, tAvgSq, lr, 0.9, 0.999, 1e-8, int32(iterInt));
        return;
    catch, end
    % 5) 全部失败 → 手写 Adam
    [net, tAvg, tAvgSq] = manual_adam_update(net, grad, tAvg, tAvgSq, lr, iterInt);
end

% ===== 手写 Adam（dlnetwork + learnables 表逐元素更新）=====
function [net, tAvg, tAvgSq] = manual_adam_update(net, grad, tAvg, tAvgSq, lr, iterInt)
    beta1 = 0.9; beta2 = 0.999; eps = 1e-8;
    t = double(iterInt);

    learn = net.Learnables;

    if isempty(tAvg)
        tAvg = learn;
        tAvg.Value = cellfun(@(v) zeros(size(v),'like',v), grad.Value, 'UniformOutput', false);
    end
    if isempty(tAvgSq)
        tAvgSq = learn;
        tAvgSq.Value = cellfun(@(v) zeros(size(v),'like',v), grad.Value, 'UniformOutput', false);
    end

    for i = 1:size(learn,1)
        g = grad.Value{i};              % dlarray
        m = beta1 * tAvg.Value{i} + (1-beta1) * g;
        v = beta2 * tAvgSq.Value{i} + (1-beta2) * (g .* g);

        mhat = m ./ (1 - beta1^t);
        vhat = v ./ (1 - beta2^t);

        step = lr .* mhat ./ (sqrt(vhat) + eps);
        learn.Value{i} = learn.Value{i} - step;

        tAvg.Value{i}   = m;
        tAvgSq.Value{i} = v;
    end

    % 写回网络参数
    net.Learnables.Value = learn.Value;
end
