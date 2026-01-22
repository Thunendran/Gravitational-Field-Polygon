% ============================================================
% Example_Triangle_Basic.m
% Demonstrates basic usage of triangle_gravity_matlab.m
% ============================================================

clear; clc;

% ------------------------------------------------------------
% Triangle geometry (3×3, rows = vertices)
% ------------------------------------------------------------
vertices = [
    0.0  0.0  0.0 ;   % v0
    1.0  0.0  0.0 ;   % v1
    0.0  1.0  0.0     % v2
];

% ------------------------------------------------------------
% Physics parameters
% ------------------------------------------------------------
G     = 1.0;
sigma = 1.0;
eps   = 1e-15;

% ------------------------------------------------------------
% Initialize triangle object
% ------------------------------------------------------------
tri = triangle_gravity_matlab('init', vertices, G, sigma, eps);

% ------------------------------------------------------------
% Observation points (N×3)
% ------------------------------------------------------------
points = [
    0.2  0.2  0.0 ;   % interior (on plane)
    0.2  0.2  1.0 ;   % above
    0.2  0.2 -1.0 ;   % below
    0.5  0.0  0.0     % mid edge
];

% ============================================================
% POTENTIAL
% ============================================================
V = triangle_gravity_matlab('potential_batch', tri, points);

% V is N×1
disp('Potential V (N×1):');
disp(V);

% ============================================================
% ACCELERATION
% ============================================================
[gx, gy, gz] = triangle_gravity_matlab('acceleration_batch', tri, points);

% gx,gy,gz are N×1
g = [gx gy gz];

disp('Acceleration g = [gx gy gz] (N×3):');
disp(g);

% ============================================================
% GRAVITY TENSOR
% ============================================================
[G11,G12,G13, ...
 G21,G22,G23, ...
 G31,G32,G33] = triangle_gravity_matlab('tensor_batch', tri, points);

% Tensor components are N×1 each
% Assemble full tensor per point if desired
N = size(points,1);
Gamma = zeros(N,3,3);
for i = 1:N
    Gamma(i,:,:) = [
        G11(i) G12(i) G13(i);
        G21(i) G22(i) G23(i);
        G31(i) G32(i) G33(i)
    ];
end

disp('Gravity tensor Γ (N×3×3, NOT symmetrized):');
disp(Gamma);

% ============================================================
% LAPLACE CHECK (EXTERIOR POINTS)
% ============================================================
lap = G11 + G22 + G33;

disp('Laplace trace (should be ~0 off the lamina):');
disp(lap);
