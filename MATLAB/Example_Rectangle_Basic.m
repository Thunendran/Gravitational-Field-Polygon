% ======================================================================
% Example_Rectangle_Basic.m
%
% Basic usage example for rectangle_gravity_matlab.m.
%
% This script demonstrates:
%   • Input and output formats
%   • Evaluation of potential, acceleration, and gravity tensor
%   • Consistent 15-digit numerical output for cross-language comparison
%
% The example is intended for:
%   • Code verification
%   • Benchmarking against Python and Julia implementations
%
% Author:
%   Thunendran Periyandy
%
% ======================================================================


clear; clc;

% ------------------------------------------------------------
% (IMPORTANT) Make sure MATLAB can see rectangle_gravity_matlab.m
% Either:
%   1) Put this example in the same folder
% or
%   2) addpath('/path/to/RectangleMATLAB');
% ------------------------------------------------------------

% ------------------------------------------------------------
% OUTPUT FORMATTER (15 significant digits, scientific notation)
% ------------------------------------------------------------
fmt = @(x) sprintf('%.15e', x);

% ------------------------------------------------------------
% INPUT FORMAT
%   points : N×3 double, rows are [x y z]
%   L, B   : half-lengths (rectangle is 2L × 2B) in plane z = z_r
%   sigma  : surface density
%   G      : gravitational constant
%   epsz   : z-regularization
%
% Tensor uses:
%   z_r      : reference plane
%   rho_surf : surface density for tensor
% ------------------------------------------------------------

% Rectangle geometry / physics (matches benchmark)
L = 1.0;
B = 1.0;
G = 1.0;

D   = 1.0;
rho = 1.0;
sigma = D * rho;

% Tensor parameters
z_r      = 0.0;
rho_surf = sigma;

% Regularization
epsz = 1e-15;

% Parallel parameters
parallel_threshold = 50000;
chunk_size         = 10000;

% ------------------------------------------------------------
% Points: N×3
% ------------------------------------------------------------
points = [
     0.0    0.0    1.0
     0.0    0.0   -1.0
     0.37  -0.41   0.83
     1.0    0.0    0.0
];

N = size(points,1);

fprintf('Number of points: %d\n\n', N);

% ------------------------------------------------------------
% 1) Potential
% Output: V is N×1
% ------------------------------------------------------------
V = rectangle_gravity_matlab('potential_batch', ...
    points, L, B, sigma, G, epsz, parallel_threshold, chunk_size);

fprintf('Potential V (N×1):\n');
for i = 1:N
    fprintf('  V[%d] = %s\n', i, fmt(V(i)));
end
fprintf('\n');

% ------------------------------------------------------------
% 2) Acceleration
% Output: gx, gy, gz are N×1
% ------------------------------------------------------------
[gx,gy,gz] = rectangle_gravity_matlab('acceleration_batch', ...
    points, L, B, sigma, G, epsz, parallel_threshold, chunk_size);

fprintf('Acceleration g (N×3):\n');
for i = 1:N
    fprintf('  g[%d] = (%s, %s, %s)\n', i, ...
        fmt(gx(i)), fmt(gy(i)), fmt(gz(i)));
end
fprintf('\n');

% ------------------------------------------------------------
% 3) Tensor
% Output:
%   Vxx, Vyy, Vzz, Vxy, Vxz, Vyz (each N×1)
% ------------------------------------------------------------
[Vxx,Vyy,Vzz,Vxy,Vxz,Vyz] = rectangle_gravity_matlab('tensor_batch', ...
    points, L, B, z_r, G, rho_surf, parallel_threshold, chunk_size);

fprintf('Gravity tensor Γ (per point, 15 digits):\n');
for i = 1:N
    fprintf('Point %d\n', i);
    fprintf('  [%s  %s  %s]\n', fmt(Vxx(i)), fmt(Vxy(i)), fmt(Vxz(i)));
    fprintf('  [%s  %s  %s]\n', fmt(Vxy(i)), fmt(Vyy(i)), fmt(Vyz(i)));
    fprintf('  [%s  %s  %s]\n\n', fmt(Vxz(i)), fmt(Vyz(i)), fmt(Vzz(i)));
end

% ------------------------------------------------------------
% Laplace trace
% ------------------------------------------------------------
lap = Vxx + Vyy + Vzz;

fprintf('Laplace trace (N×1):\n');
for i = 1:N
    fprintf('  Lap[%d] = %s\n', i, fmt(lap(i)));
end
