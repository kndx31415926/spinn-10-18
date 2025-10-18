function spinn_train_model_v2_dyn(data_path, hnn_model_path, out_model_path, opts)
% spinn_train_model_v2_dyn — 按推理口径（动态特征）重训 v2 学生网络
% 目标：学到 w(t) = f([params25, F10(q(t),dq(t),qT,...)])
%
% 输入：
%   data_path       : 训练集参数矩阵 .mat（含 params_matrix_spinn 或 params_matrix）
%   hnn_model_path  : HNN 模型 .mat（含 dlnet, muX, sigmaX）
%   out_model_path  : 输出模型 .mat（默认 'trained_model_spinn_v2.mat'）
%   opts            : 结构体（可选），关键字段：
%       .teacher           = 'nn_v1' | 'ratio' | 'equal'（默认 'nn_v1'）
%       .teacher_model_v1  = 'trained_model_spinn.mat'（teacher=nn_v1 时用）
%       .useGPU            = []（自动）
%       .seed              = 42
%       .max_rows          = inf（从数据表取前 N 行）
%       .dt                = 0.002
%       .t_final           = 6.0
%       .sample_stride     = 5     （每隔多少步采一个样本）
%       .radius            = 0.03  （命中阈值，命中后可提前结束此条滚动）
%       .k_dir_max         = 25    （与 v1 一致）
%       .omega_eps         = 1e-3
%       .fill_power        = true
%       .fill_alpha        = 0.90
%       .p_boost_max       = 5.0
%       .w_floor           = 0.05
%       .ou_tau            = 0.30  （OU 阻尼时间常数；与 v1 一致）
%       .val_ratio         = 0.15
%       .epochs            = 120
%       .mb                = 256
%       .lr                = 1e-3
%
% 依赖：
%   - computeDynamics.m
%   - MATLAB Deep Learning Toolbox（dlnetwork/dlarray/dlfeval）
%
% 产物（out_model_path）：
%   trainedNet, muX, sigmaX, muY, sigmaY, feature_names, label_names, meta
%
% 2025 · kndx/spinn

    % -------- 0) 入参与默认 --------
    if nargin < 1 || isempty(data_path)
        data_path = fullfile('C:','Users','kndx9','Desktop','SpinnMechanicalArmParams.mat');
    end
    if nargin < 2 || isempty(hnn_model_path)
        hnn_model_path = 'spinn_hnn_model.mat';
    end
    if nargin < 3 || isempty(out_model_path)
        out_model_path = 'trained_model_spinn_v2.mat';
    end
    if nargin < 4, opts = struct(); end

    def = struct();
    def.teacher          = 'nn_v1';
    def.teacher_model_v1 = 'trained_model_spinn.mat';
    def.useGPU           = [];
    def.seed             = 42;
    def.max_rows         = inf;

    def.dt               = 0.002;
    def.t_final          = 6.0;
    def.sample_stride    = 5;
    def.radius           = 0.03;

    def.k_dir_max        = 25;
    def.omega_eps        = 1e-3;
    def.fill_power       = true;
    def.fill_alpha       = 0.90;
    def.p_boost_max      = 5.0;
    def.w_floor          = 0.05;
    def.ou_tau           = 0.30;

    def.val_ratio        = 0.15;
    def.epochs           = 120;
    def.mb               = 256;
    def.lr               = 1e-3;

    opts = merge_opts(def, opts);
    if isempty(opts.useGPU)
        if exist('canUseGPU','file')==2, opts.useGPU = canUseGPU(); else, opts.useGPU = false; end
    end
    if ~isempty(opts.seed), rng(double(opts.seed)); end

    % -------- 1) 载入参数表 --------
    S = load(data_path);
    if isfield(S,'params_matrix_spinn')
        PM = S.params_matrix_spinn;
    elseif isfield(S,'params_matrix')
        PM = S.params_matrix;
    else
        error('在 %s 中未找到 params_matrix[_spinn]。', data_path);
    end
    if size(PM,2) < 29
        error('数据表列数不足（需要至少 29 列，含 25 输入 + 4 标签）。');
    end
    PM = PM(:,1:29);
    PM = PM(all(isfinite(PM),2), :);
    if isfinite(opts.max_rows)
        PM = PM(1:min(opts.max_rows, size(PM,1)), :);
    end

    % -------- 2) 载入 HNN 模型（用于 F10） --------
    H = load(hnn_model_path);
    if isfield(H,'model') && isstruct(H.model) && isfield(H.model,'dlnet')
        dlnetH = H.model.dlnet;  muH = H.model.muX;  sigH = H.model.sigmaX;
    else
        dlnetH = H.dlnet;        muH = H.muX;        sigH = H.sigmaX;
    end
    assert(~isempty(dlnetH) && ~isempty(muH) && ~isempty(sigH), 'HNN 模型缺字段。');
    muHv = dlarray(single(muH(:)), 'CB');  sigHv = dlarray(single(sigH(:)), 'CB');
    if opts.useGPU, muHv=gpuArray(muHv); sigHv=gpuArray(sigHv); end

    % -------- 3) 载入 v1 教师模型（可选） --------
    T = struct();  hasV1 = false;
    if strcmpi(opts.teacher,'nn_v1')
        if isfile(opts.teacher_model_v1)
            T = load(opts.teacher_model_v1);
            need = {'trainedNet','muX','sigmaX','muY','sigmaY'};
            hasV1 = all(isfield(T, need));
            if hasV1
                T.muX = T.muX(:).'; T.sigmaX = T.sigmaX(:).';
                T.muY = T.muY(:).'; T.sigmaY = T.sigmaY(:).';
            end
        end
        if ~hasV1
            warning('找不到或不完整的 v1 教师模型，teacher 回退为 ratio。');
            opts.teacher = 'ratio';
        end
    end

    % -------- 4) 滚动仿真收集样本（动态特征） --------
    X_all = [];   % N x 35
    Y_all = [];   % N x 3  (w)
    L = [0.24 0.214 0.324]; g = 9.81;      % 固定几何
    fprintf('收集动态样本...\n');

    for i = 1:size(PM,1)
        p25 = PM(i,1:25);
        % 解包（与推理一致）
        m        = p25(1:3);
        dq       = p25(4:6).';
        dampNom  = p25([7 9 11]);
        zetaNom  = p25([8 10 12]);
        q0d      = p25([13 14 15]);
        qTd      = p25([16 17 18]);
        Pmax     = p25(22);
        Prated   = p25([23 24 25]);

        q  = deg2rad(q0d(:));
        qT = deg2rad(qTd(:));

        tvec = (0:opts.dt:opts.t_final).';
        xi = zeros(3,1);   % OU 阻尼状态

        for k = 1:numel(tvec)
            % —— 4.1 动态特征（按推理口径）——
            F10 = build_features_v2(q, dq, qT, m, L, g, dlnetH, muHv, sigHv, opts.useGPU);
            X = [p25, F10];    % 25 + 10 = 35

            % —— 4.2 教师权重 w_teacher(t) —— 
            w_teacher = teacher_w(opts.teacher, q, dq, qT, p25, Pmax, Prated, ...
                                  L, opts, T, opts.w_floor);

            % 采样（降采样）
            if mod(k-1, opts.sample_stride) == 0
                X_all = [X_all; X]; %#ok<AGROW>
                Y_all = [Y_all; w_teacher]; %#ok<AGROW>
            end

            % —— 4.3 用“与推理一致”的控制律推进（几何最速 + fill-power + 双顶帽）——
            [xE,yE] = fk_end(q, L);
            [xT,yT] = fk_end(qT, L);
            r = [xT - xE; yT - yE]; nr = norm(r); if nr>0, r = r/nr; else, r=[0;0]; end
            J = jacobian_planar(q, L);
            tau_dir = J.' * r;
            nd = norm(tau_dir); if nd>0, tau_dir = tau_dir/nd; end
            tau_des = opts.k_dir_max * tau_dir;

            omega   = dq;
            omega_nz = omega;
            mask = abs(omega_nz) < opts.omega_eps;
            omega_nz(mask) = opts.omega_eps .* sign(omega_nz(mask)) + opts.omega_eps .* (omega_nz(mask)==0);

            p_des = tau_des .* omega_nz;

            if opts.fill_power
                s_now = sum(abs(p_des));
                if s_now > 0 && s_now < opts.fill_alpha * Pmax
                    scale = min(opts.p_boost_max, Pmax / s_now);
                    p_des = p_des * scale;
                end
            end

            % 先总功率限幅
            s_tot = sum(abs(p_des));
            Pmax_eff = Pmax;
            if s_tot > Pmax_eff && s_tot > 0
                p_mid = p_des * (Pmax_eff / s_tot);
            else
                p_mid = p_des;
            end
            % 再分轴硬帽
            cap_i = min(w_teacher(:) * Pmax_eff, Prated(:));
            p_cap = sign(p_mid) .* min(abs(p_mid), cap_i);
            tau_cmd = p_cap ./ omega_nz;

            % 动力学（含 OU 阻尼）
            [Mq,Cq,Gq] = computeDynamics(m(1),m(2),m(3), L(1),L(2),L(3), g, q, dq);
            xi = xi + (-xi/opts.ou_tau)*opts.dt + sqrt(2*opts.dt/opts.ou_tau)*randn(3,1);
            D = diag(max(0, dampNom(:) .* (1 + zetaNom(:) .* xi)));
            ddq = (Mq + 1e-6*eye(3)) \ (tau_cmd - Cq*dq - Gq - D*dq);

            dq = dq + ddq*opts.dt;
            q  = q  + dq *opts.dt;

            % 命中提前结束此样本的滚动
            dEE = hypot(xE - xT, yE - yT);
            if dEE <= opts.radius, break; end
        end

        if mod(i,10)==0
            fprintf('  已处理 %d / %d 条参数\n', i, size(PM,1));
        end
    end

    % -------- 5) 标准化、划分、训练 --------
    feature_names = feature_names_v2();    % 25 + 10
    if size(X_all,2) ~= numel(feature_names)
        error('特征维度不一致：X=%d, names=%d', size(X_all,2), numel(feature_names));
    end

    [Xz, muX, sigmaX] = zscore(X_all);
    [Yz, muY, sigmaY] = zscore(Y_all);
    sigmaX(sigmaX<1e-6) = 1e-6;
    sigmaY(sigmaY<1e-6) = 1e-6;

    N = size(Xz,1);
    idx = randperm(N);
    nva = round(opts.val_ratio * N);
    id_va = idx(1:nva); id_tr = idx(nva+1:end);

    Xtr = Xz(id_tr,:); Ytr = Yz(id_tr,:);
    Xva = Xz(id_va,:); Yva = Yz(id_va,:);

    inDim = size(Xtr,2);
    layers = [
        featureInputLayer(inDim,'Normalization','none','Name','in')
        fullyConnectedLayer(256,'Name','fc1')
        reluLayer('Name','r1')
        fullyConnectedLayer(128,'Name','fc2')
        reluLayer('Name','r2')
        fullyConnectedLayer(64,'Name','fc3')
        reluLayer('Name','r3')
        fullyConnectedLayer(3,'Name','out')      % 只回归 3 维 w
        regressionLayer('Name','reg')
    ];
    options = trainingOptions('adam', ...
        'MaxEpochs', opts.epochs, ...
        'InitialLearnRate', opts.lr, ...
        'MiniBatchSize', opts.mb, ...
        'ValidationData', {Xva, Yva}, ...
        'ValidationFrequency', 50, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'Verbose', true, ...
        'L2Regularization', 1e-4);

    fprintf('开始训练：样本 %d，输入维度 %d\n', size(Xtr,1)+size(Xva,1), inDim);
    trainedNet = trainNetwork(Xtr, Ytr, layers, options);

    % 验证误差（去标准化）
    Yva_pred = predict(trainedNet, Xva);
    Yva_pred = bsxfun(@times, Yva_pred, sigmaY) + muY;
    Yva_true = bsxfun(@times, Yva,      sigmaY) + muY;
    mse_w  = mean((Yva_pred - Yva_true).^2, 'all');
    fprintf('Val MSE(w): %.6f | 样本=%d | inDim=%d\n', mse_w, size(Xtr,1)+size(Xva,1), inDim);

    % -------- 6) 保存模型 --------
    label_names = {'w1','w2','w3'};
    meta = struct();
    meta.version = 'v2-dyn';
    meta.teacher = opts.teacher;
    if strcmpi(opts.teacher,'nn_v1'), meta.teacher_model_v1 = opts.teacher_model_v1; end
    meta.hnn = struct('model_path', hnn_model_path, 'mu', muH, 'sigma', sigH);
    meta.train_opts = opts;

    save(out_model_path, 'trainedNet','muX','sigmaX','muY','sigmaY', ...
         'feature_names','label_names','meta');
    fprintf('已保存：%s\n', out_model_path);
end

% =============== 内部工具 ===============

function names = feature_names_v2()
    names = { ...
        'm1','m2','m3', ...                        % 1-3
        'dq0_1','dq0_2','dq0_3', ...               % 4-6
        'damp1','zeta1','damp2','zeta2','damp3','zeta3', ... % 7-12
        'init_deg1','init_deg2','init_deg3', ...   % 13-15
        'tgt_deg1','tgt_deg2','tgt_deg3', ...      % 16-18
        'dtheta1','dtheta2','dtheta3', ...         % 19-21
        'Pmax', ...                                 % 22
        'Prated1','Prated2','Prated3', ...         % 23-25
        'dEE','M11','M22','M33','condM','H0','Hg','DeltaH','norm_dHdp','norm_dHdq' ... % 26-35
    };
end

function F10 = build_features_v2(q, dq, qT, m, L, g, dlnetH, muHv, sigHv, useGPU)
    % F10 = [ dEE, M11 M22 M33, condM, H0, Hg, DeltaH, ||dHdp||, ||dHdq|| ]
    [Mq,~,~] = computeDynamics(m(1),m(2),m(3), L(1),L(2),L(3), g, q, dq);
    p = Mq * dq;
    [xE,yE] = fk_end(q,  L);
    [xT,yT] = fk_end(qT, L);
    dEE = hypot(xE-xT, yE-yT);
    rc = rcond(Mq); condM = max(1, 1/max(rc, 1e-15));

    % HNN 能量与梯度（dlfeval + 被跟踪的 dlarray）
    qv = dlarray(single(q(:)),'CB'); pv = dlarray(single(p(:)),'CB');
    if useGPU, qv=gpuArray(qv); pv=gpuArray(pv); end
    [H0dl,dHdq0dl,dHdp0dl] = dlfeval(@hnn_grad_fun, dlnetH, qv, pv, muHv, sigHv);
    [Hgdl,~,~] = dlfeval(@hnn_grad_fun, dlnetH, ...
                         dlarray(single(qT(:)),'CB'), dlarray(single(zeros(3,1)),'CB'), ...
                         muHv, sigHv);
    H0 = double(gather(extractdata(H0dl)));
    Hg = double(gather(extractdata(Hgdl)));
    dHdp_norm = double(gather(norm(extractdata(dHdp0dl))));
    dHdq_norm = double(gather(norm(extractdata(dHdq0dl))));
    DeltaH = H0 - Hg;

    F10 = [dEE, diag(Mq).', condM, H0, Hg, DeltaH, dHdp_norm, dHdq_norm];
end

function [H,dHdq,dHdp] = hnn_grad_fun(dlnet, qv, pv, muHv, sigHv)
    z  = ([qv; pv] - muHv) ./ sigHv;
    H  = forward(dlnet, z);
    Hs = sum(H,'all');                  % 标量
    [dHdq,dHdp] = dlgradient(Hs, qv, pv);
end

function w = teacher_w(mode, q, dq, qT, p25, Pmax, Prated, L, opts, T, w_floor)
    switch lower(mode)
        case 'nn_v1'
            w = predict_w_v1(q, dq, qT, p25, Pmax, Prated, T, w_floor);
        case 'ratio'
            [xE,yE] = fk_end(q, L);    [xT,yT] = fk_end(qT, L);
            r = [xT-xE; yT-yE]; nr = norm(r); if nr>0, r=r/nr; else, r=[0;0]; end
            J = jacobian_planar(q, L);
            tau_dir = J.' * r; nd = norm(tau_dir); if nd>0, tau_dir=tau_dir/nd; end
            tau_des = opts.k_dir_max * tau_dir;
            omega = dq;
            omega(abs(omega)<opts.omega_eps) = opts.omega_eps;
            p_des = abs(tau_des .* omega);
            s = sum(p_des);
            if s>0, w = (p_des(:).'/s); else, w = [1 1 1]/3; end
            w = project_to_simplex_floor(w, w_floor);
        otherwise
            w = [1 1 1]/3;
    end
end

function w = predict_w_v1(q_now, dq_now, qT, p25, Pmax, Prated, T, w_floor)
    % v1 教师：按 v1 特征口径在线重算（init=当前角度，tgt=目标，dtheta=差）
    init_deg_now = rad2deg(q_now(:)).';
    qT_deg = rad2deg(qT(:)).';
    dth_deg_now = qT_deg - init_deg_now;
    x = [ p25(1:3), dq_now(:).', p25(7:12), init_deg_now, qT_deg, dth_deg_now, Pmax, Prated ];
    xz = (x - T.muX) ./ T.sigmaX;
    yp = predict(T.trainedNet, xz);
    yp = yp .* T.sigmaY + T.muY;
    w  = project_to_simplex_floor(yp(1:3), w_floor);
end

function [x3,y3] = fk_end(q, L)
    q1=q(1); q2=q(2); q3=q(3);
    x1=L(1)*cos(q1);            y1=L(1)*sin(q1);
    x2=x1+L(2)*cos(q1+q2);      y2=y1+L(2)*sin(q1+q2);
    x3=x2+L(3)*cos(q1+q2+q3);   y3=y2+L(3)*sin(q1+q2+q3);
end

function J = jacobian_planar(q, L)
    q1=q(1); q2=q(2); q3=q(3);
    J = [ -L(1)*sin(q1)-L(2)*sin(q1+q2)-L(3)*sin(q1+q2+q3),  -L(2)*sin(q1+q2)-L(3)*sin(q1+q2+q3),  -L(3)*sin(q1+q2+q3);
           L(1)*cos(q1)+L(2)*cos(q1+q2)+L(3)*cos(q1+q2+q3),   L(2)*cos(q1+q2)+L(3)*cos(q1+q2+q3),    L(3)*cos(q1+q2+q3) ];
end

function w = project_to_simplex_floor(w, floor_val)
    w = double(w(:).'); floor_val = max(0, min(1/3 - 1e-6, floor_val));
    free = 1 - 3*floor_val;
    w = max(w,0); s=sum(w); if s==0, w=ones(size(w))/numel(w); else, w=w/s; end
    w = max(w - floor_val, 0); s2=sum(w); if s2>0, w=w/s2*free; end
    w = w + floor_val;
end

function M = merge_opts(M, U)
    if nargin<2 || ~isstruct(U), return; end
    f=fieldnames(U); for i=1:numel(f), M.(f{i})=U.(f{i}); end
end
