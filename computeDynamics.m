function [M, C, G] = computeDynamics(m1, m2, m3, L1, L2, L3, g, q, dq)
% computeDynamics 3R 平面臂动力学：惯性矩阵 M(q)、科氏/离心项 C(q,dq)、重力项 G(q)
% 接口与现有工程保持一致：
%   输入:
%     m1..m3 : 三段连杆质量
%     L1..L3 : 三段连杆长度
%     g      : 重力加速度（正值，单位 m/s^2）
%     q      : 3x1 关节角 (rad)
%     dq     : 3x1 关节角速度 (rad/s)
%   输出:
%     M(3x3) : 惯性矩阵（对称正定）
%     C(3x3) : 使得 tau_coriolis = C(q,dq) * dq
%     G(3x1) : 重力广义力向量（下垂方向沿 -y；这里按 V=Σ m g y 构造）

    % -------- 基本量 --------
    q = q(:); dq = dq(:);
    q1 = q(1); q2 = q(2); q3 = q(3);

    % 质心位置（相对各段近端关节的中点）
    r1 = L1/2; r2 = L2/2; r3 = L3/2;

    % 每段关于“自身质心”绕 z 的转动惯量（细长杆模型）
    I1 = m1*L1^2/12;
    I2 = m2*L2^2/12;
    I3 = m3*L3^2/12;

    % -------- 各段 COM 位置与雅可比 --------
    % 端点/合成角
    c1 = cos(q1); s1 = sin(q1);
    c12 = cos(q1+q2); s12 = sin(q1+q2);
    c123 = cos(q1+q2+q3); s123 = sin(q1+q2+q3);

    % COM 位置（用于重力）
    p1 = [ r1*c1;                           r1*s1                           ];
    p2 = [ L1*c1 + r2*c12;                  L1*s1 + r2*s12                  ];
    p3 = [ L1*c1 + L2*c12 + r3*c123;        L1*s1 + L2*s12 + r3*s123        ];

    % 线速度雅可比 Jv_i (2x3)
    Jv1 = [ -r1*s1,                      0,                  0;
             r1*c1,                      0,                  0 ];

    Jv2 = [ -L1*s1 - r2*s12,            -r2*s12,            0;
             L1*c1 + r2*c12,             r2*c12,            0 ];

    Jv3 = [ -L1*s1 - L2*s12 - r3*s123,  -L2*s12 - r3*s123,  -r3*s123;
             L1*c1 + L2*c12 + r3*c123,   L2*c12 + r3*c123,   r3*c123 ];

    % 角速度雅可比 Jw_i (1x3，z 轴)
    Jw1 = [1 0 0];       % link1 受 q1
    Jw2 = [1 1 0];       % link2 受 q1,q2
    Jw3 = [1 1 1];       % link3 受 q1,q2,q3

    % -------- M(q)：能量/雅可比法（绝不双算平行轴）--------
    M = zeros(3,3);
    M = M + m1*(Jv1.'*Jv1) + (Jw1.'*Jw1)*I1;
    M = M + m2*(Jv2.'*Jv2) + (Jw2.'*Jw2)*I2;
    M = M + m3*(Jv3.'*Jv3) + (Jw3.'*Jw3)*I3;

    % 对称化，数值稳健
    M = 0.5*(M + M.');

    % -------- G(q)：重力广义力（梯度法）--------
    % V = Σ m_i g y_i  →  G = ∂V/∂q
    % 直接用 Jv 的第 2 行（对 y 的导数）
    dVy = m1*g*Jv1(2,:) + m2*g*Jv2(2,:) + m3*g*Jv3(2,:);
    G = dVy.';   % 3x1

    % -------- C(q,dq)：Christoffel 数值构造（C*dq 给科氏/离心力）--------
    % 若存在用户自定义的解析 C 函数，可优先使用
    try
        if exist('Cq_function','file')==2
            C = Cq_function(q, dq);
            % 保险：尺寸检查
            if ~isequal(size(C), [3 3]), error('badC'); end
            return;
        end
    catch
        % 回退到数值构造
    end

    % 数值差分 ∂M/∂q_k
    dM = zeros(3,3,3);
    for k = 1:3
        h = max(1e-6, 1e-6*(1+abs(q(k))));
        qp = q; qm = q;
        qp(k) = qp(k) + h;
        qm(k) = qm(k) - h;
        Mp = local_M(qp, m1,m2,m3, L1,L2,L3);
        Mm = local_M(qm, m1,m2,m3, L1,L2,L3);
        dM(:,:,k) = (Mp - Mm) / (2*h);
    end

    % Christoffel: c_ijk = 0.5*(dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
    % C(i,j) = Σ_k c_ijk * dq_k
    C = zeros(3,3);
    for i = 1:3
        for j = 1:3
            cij = 0;
            for k = 1:3
                cijk = 0.5 * ( dM(i,j,k) + dM(i,k,j) - dM(j,k,i) );
                cij = cij + cijk * dq(k);
            end
            C(i,j) = cij;
        end
    end
end

% ===== 局部：仅计算 M(q)（供数值差分用；与主函数同口径）=====
function M = local_M(q, m1,m2,m3, L1,L2,L3)
    q1 = q(1); q2 = q(2); q3 = q(3);

    r1 = L1/2; r2 = L2/2; r3 = L3/2;
    I1 = m1*L1^2/12; I2 = m2*L2^2/12; I3 = m3*L3^2/12;

    c1 = cos(q1); s1 = sin(q1);
    c12 = cos(q1+q2); s12 = sin(q1+q2);
    c123 = cos(q1+q2+q3); s123 = sin(q1+q2+q3);

    Jv1 = [ -r1*s1,                      0,                  0;
             r1*c1,                      0,                  0 ];

    Jv2 = [ -L1*s1 - r2*s12,            -r2*s12,            0;
             L1*c1 + r2*c12,             r2*c12,            0 ];

    Jv3 = [ -L1*s1 - L2*s12 - r3*s123,  -L2*s12 - r3*s123,  -r3*s123;
             L1*c1 + L2*c12 + r3*c123,   L2*c12 + r3*c123,   r3*c123 ];

    Jw1 = [1 0 0];
    Jw2 = [1 1 0];
    Jw3 = [1 1 1];

    M = zeros(3,3);
    M = M + m1*(Jv1.'*Jv1) + (Jw1.'*Jw1)*I1;
    M = M + m2*(Jv2.'*Jv2) + (Jw2.'*Jw2)*I2;
    M = M + m3*(Jv3.'*Jv3) + (Jw3.'*Jw3)*I3;
    M = 0.5*(M + M.');
end
