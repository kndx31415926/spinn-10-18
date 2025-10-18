function [tau_out, q_out, dq_out, info] = spinn_joint_hardlimit(mode, q, dq, tau_in, cfg)
% spinn_joint_hardlimit
% 3R 平面臂“硬限位”约束（不允许越界，不反弹）：
% - mode='pre' : 功率/力矩计算前调用，消去沿越界方向的推进力矩（防继续往外推）
% - mode='post': 推进得到 q_next,dq_next 后调用，夹断 q 并将指向越界外的速度置零（塑性冲击，e=0）
%
% 用法（每步两处，示例见本文末“接入位置”）：
%   % --- pre：算 p_des=tau.*omega 之前 ---
%   [tau_des, ~, ~] = spinn_joint_hardlimit('pre', q, dq_used_for_power, tau_des, joint_cfg);
%   % --- post：更新 q_next,dq_next 之后 ---
%   [~, q_next, dq_next] = spinn_joint_hardlimit('post', q_next, dq_next, [], joint_cfg);
%
% cfg 字段（全部可选）：
%   .qmin_deg = [-175 -5  -5 ];   % 关节下界（度）
%   .qmax_deg = [ 175 175 175];   % 关节上界（度）
%   .deadband_deg = 0.5;          % 边界死区（度），避免数值颤动
%   .freeze_inward = true;        % pre 阶段：越界方向的 tau 置零
%   .zero_vel_on_contact = true;  % post 阶段：贴边且速度指向外侧→置 0
%
% 返回：
%   tau_out : 约束后的力矩（pre 有效；post 原样返回 []）
%   q_out   : 约束后的角度（post 有效；pre 原样返回 q）
%   dq_out  : 约束后的角速度（post 有效；pre 原样返回 dq）
%   info    : 诊断（atLower/atUpper/clamped 等）
%
% 说明：
% - 本函数只处理“**硬边界**”几何约束，不改动你的动力学核 computeDynamics、
%   双顶帽功率限幅或辛式积分的口径；与 PID/NN/H_θ 三条链均兼容。 
% - 通过“pre 去力 + post 投影”的组合，可避免“弹开”引起的乱跳（恢复系数 e=0）。
%
% (c) SPINN stack

    narginchk(5,5);
    mode = lower(string(mode));
    q = q(:); dq = dq(:);
    if ~isempty(tau_in), tau_in = tau_in(:); end
    assert(numel(q)==3 && numel(dq)==3, 'q,dq 必须为 3x1。');

    % ===== 默认参数 =====
    C = local_defaults();
    C = local_merge(C, cfg);

    qmin = deg2rad(C.qmin_deg(:));
    qmax = deg2rad(C.qmax_deg(:));
    epsA = deg2rad(C.deadband_deg);

    atLower = q <= (qmin + epsA);
    atUpper = q >= (qmax - epsA);

    switch mode
        case "pre"
            tau_out = tau_in; q_out = q; dq_out = dq;
            if C.freeze_inward
                % 下边界：禁止继续向负方向推（τ<0）；上边界：禁止继续向正方向推（τ>0）
                maskL = atLower & (tau_out < 0);
                maskU = atUpper & (tau_out > 0);
                tau_out(maskL | maskU) = 0;
            else
                tau_out = tau_in;
            end
            info = struct('mode','pre','atLower',atLower,'atUpper',atUpper);

        case "post"
            % 1) 角度硬投影
            q_proj = min(max(q, qmin), qmax);

            % 2) 速度塑性冲击（不反弹 e=0）：贴边且速度指向边界外 → 置零
            dq_proj = dq;
            if C.zero_vel_on_contact
                atL_now = q_proj <= (qmin + epsA);
                atU_now = q_proj >= (qmax - epsA);
                % 下边界：负向速度 → 置零；上边界：正向速度 → 置零
                dq_proj(atL_now & (dq_proj < 0)) = 0;
                dq_proj(atU_now & (dq_proj > 0)) = 0;
            end

            tau_out = []; q_out = q_proj; dq_out = dq_proj;
            info = struct('mode','post','clamped', (q_proj~=q), ...
                          'atLower', q_proj <= (qmin + epsA), ...
                          'atUpper', q_proj >= (qmax - epsA));
        otherwise
            error('mode 只能为 "pre" 或 "post"。');
    end
end

% ===== 默认配置 =====
function C = local_defaults()
    C.qmin_deg = [-175 -175 -175];
    C.qmax_deg = [ 175  175  175];
    C.deadband_deg = 0.5;
    C.freeze_inward = true;
    C.zero_vel_on_contact = true;
end

% ===== 结构体浅合并 =====
function O = local_merge(O, U)
    if isempty(U) || ~isstruct(U), return; end
    f = fieldnames(U);
    for i=1:numel(f), O.(f{i}) = U.(f{i}); end
end
