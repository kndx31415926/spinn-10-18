function spinn_train_model(data_path, out_model_path, var_name)
% spinn_train_model
% 训练 SPINN 版功率分配网络：
%  - 输入 X：前 25 列
%  - 标签 y：后 4 列（[w1,w2,w3,t_hit]）
%  - 标准化：对 X、y 分别 zscore，并落盘 muX/sigmaX/muY/sigmaY
%  - 保存：trained_model_spinn.mat（含网络与标准化参数）
%
% 用法（默认路径/变量名）：
%   spinn_train_model();
% 自定义：
%   spinn_train_model('C:\Users\kndx9\Desktop\SpinnMechanicalArmParams.mat', ...
%                     'trained_model_spinn.mat', 'params_matrix_spinn');

    % -------------------- 1) 参数与数据载入 --------------------
    if nargin < 1 || isempty(data_path)
        % 与数据生成默认输出保持一致（SpinnMechanicalArmParams.mat）
        data_path = 'C:\Users\kndx9\Desktop\SpinnMechanicalArmParams.mat';
    end
    if nargin < 2 || isempty(out_model_path)
        out_model_path = 'trained_model_spinn.mat';
    end
    if nargin < 3 || isempty(var_name)
        var_name = 'params_matrix_spinn';   % 首选 SPINN 版变量名
    end

    S = load(data_path);
    if isfield(S, var_name)
        PM = S.(var_name);
    elseif isfield(S, 'params_matrix_spinn')
        PM = S.params_matrix_spinn;
    elseif isfield(S, 'params_matrix')
        % 兼容老变量名（我们已在生成阶段并写两份，因此两者都可）
        PM = S.params_matrix;
        warning('未找到 %s，已回退到 params_matrix。', var_name);
    else
        error('在 %s 中未找到训练矩阵变量（params_matrix[_spinn]）。', data_path);
    end

    % 基本健壮性：仅保留前 29 列；去 NaN/Inf
    if size(PM,2) < 29
        error('数据列数不足：需要 29 列（25 输入 + 4 标签），实际为 %d。', size(PM,2));
    end
    PM = PM(:, 1:29);
    bad = any(~isfinite(PM), 2);
    PM = PM(~bad, :);

    % -------------------- 2) 切割与标签修正 --------------------
    X = PM(:, 1:25);
    y = PM(:, 26:29);     % [w1,w2,w3,t_hit]

    % 将前三维权重投影到单纯形：非负且和=1（老师优化正常情况下已满足，这里保险）
    y(:,1:3) = project_to_simplex_rows(y(:,1:3));

    % -------------------- 3) 标准化 --------------------
    [X, muX, sigmaX] = zscore(X);
    [y, muY, sigmaY] = zscore(y);
    sigmaX(sigmaX < 1e-6) = 1e-6;
    sigmaY(sigmaY < 1e-6) = 1e-6;

    % -------------------- 4) 数据集划分 --------------------
    num_samples = size(X, 1);
    rng(42); 
    idx = randperm(num_samples);
    train_ratio = 0.70; 
    val_ratio   = 0.15;
    ntr = round(train_ratio * num_samples);
    nva = round(val_ratio   * num_samples);

    id_tr = idx(1:ntr);
    id_va = idx(ntr+1:ntr+nva);
    id_te = idx(ntr+nva+1:end);

    Xtr = X(id_tr,:);  ytr = y(id_tr,:);
    Xva = X(id_va,:);  yva = y(id_va,:);
    Xte = X(id_te,:);  yte = y(id_te,:);

    % -------------------- 5) 网络结构 --------------------
    layers = [
        featureInputLayer(25, 'Normalization','none','Name','in')
        fullyConnectedLayer(128,'Name','fc1')
        reluLayer('Name','r1')
        fullyConnectedLayer(128,'Name','fc2')
        reluLayer('Name','r2')
        fullyConnectedLayer(64,'Name','fc3')
        reluLayer('Name','r3')
        fullyConnectedLayer(4,'Name','out')
        regressionLayer('Name','reg')
    ];

    options = trainingOptions('adam', ...
        'MaxEpochs', 200, ...
        'InitialLearnRate', 1e-4, ...
        'MiniBatchSize', 32, ...
        'ValidationData', {Xva, yva}, ...
        'ValidationFrequency', 30, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'Verbose', true, ...
        'L2Regularization', 1e-4);

    % -------------------- 6) 训练 --------------------
    trainedNet = trainNetwork(Xtr, ytr, layers, options);

    % -------------------- 7) 评估（反标准化后给直观指标） --------------------
    y_pred_te = predict(trainedNet, Xte);

    % 反标准化
    y_pred_te = bsxfun(@times, y_pred_te, sigmaY) + muY;
    y_te_true = bsxfun(@times, yte,      sigmaY) + muY;

    % 将前三维投影到单纯形（仅用于评估观察；推理时也会这样做）
    w_pred_proj = project_to_simplex_rows(y_pred_te(:,1:3));
    w_true      = y_te_true(:,1:3);

    mse_w  = mean((w_pred_proj - w_true).^2,'all');
    mse_t  = mean((y_pred_te(:,4) - y_te_true(:,4)).^2);
    mse_all= mean(([w_pred_proj, y_pred_te(:,4)] - [w_true, y_te_true(:,4)]).^2,'all');

    fprintf('Test MSE (weights 3-dim): %.6f\n', mse_w);
    fprintf('Test MSE (time 1-dim)   : %.6f\n', mse_t);
    fprintf('Test MSE (overall 4-dim): %.6f\n', mse_all);

    % 简单的“和为1”偏差报告（未投影的原生输出）
    sum_violation = mean(abs(sum(y_pred_te(:,1:3),2) - 1));
    fprintf('Mean |sum(w_pred_raw)-1| on test: %.4f\n', sum_violation);

    % -------------------- 8) 落盘（供推理/仿真使用） --------------------
    feature_names = spinn_feature_names();   % 25 维输入字段名提示
    label_names   = {'w1','w2','w3','t_hit'};

    save(out_model_path, 'trainedNet','muX','sigmaX','muY','sigmaY', ...
         'feature_names','label_names');

    fprintf('SPINN 模型与标准化参数已保存到：%s\n', out_model_path);
end

% ===== 辅助：逐行投影到单纯形（每行非负且和为1） =====
function W = project_to_simplex_rows(W)
    W = max(W, 0);
    s = sum(W,2);
    s(s==0) = 1;
    W = W ./ s;
end

% ===== 仅用于记录：25 维输入字段名（与采样器一致）=====
function names = spinn_feature_names()
    names = { ...
        'm1','m2','m3', ...                        % 1-3
        'dq0_1','dq0_2','dq0_3', ...               % 4-6
        'damp1','zeta1','damp2','zeta2','damp3','zeta3', ... % 7-12
        'init_deg1','init_deg2','init_deg3', ...   % 13-15
        'tgt_deg1','tgt_deg2','tgt_deg3', ...      % 16-18
        'dtheta1','dtheta2','dtheta3', ...         % 19-21
        'Pmax', ...                                 % 22
        'Prated1','Prated2','Prated3' };           % 23-25
end
