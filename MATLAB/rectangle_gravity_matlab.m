% ======================================================================
% rectangle_gravity_matlab.m
%
% Analytical gravitational field of a homogeneous planar rectangle
% using closed-form expressions.
%
% This code computes:
%   • Gravitational potential
%   • Gravitational acceleration (first spatial derivatives)
%   • Gravity-gradient tensor (second spatial derivatives)
%
% The implementation is mathematically equivalent to the Python and Julia
% reference codes and follows the same corner-summation conventions,
% sign rules, and safe-math regularization for logarithmic and arctangent
% singularities.
%
% Features:
%   • Vectorized batch evaluation for N×3 input points
%   • Optional parallel execution using parfor
%   • Machine-precision agreement with Python and Julia implementations
%   • Deterministic handling of edge and vertex singularities
%
% Input format:
%   points : N×3 array, rows are (x, y, z)
%   Geometry defined by half-lengths L and B (rectangle size = 2L × 2B)
%
% Output format:
%   Potential        : N×1
%   Acceleration     : N×1 vectors (gx, gy, gz)
%   Gravity tensor   : N×1 components (Vxx, Vyy, Vzz, Vxy, Vxz, Vyz)
%
% Author:
%   Thunendran Periyandy
%
% Created for:
%
% ======================================================================


function varargout = rectangle_gravity_matlab(action, varargin)
% Public dispatcher for rectangle gravity routines in this file.
% Usage:
%   V = rectangle_gravity_matlab('potential_batch', points, L, B, sigma, G, epsz, parallel_threshold, chunk_size)
%   [gx,gy,gz] = rectangle_gravity_matlab('acceleration_batch', points, L, B, sigma, G, epsz, parallel_threshold, chunk_size)
%   [Vxx,Vyy,Vzz,Vxy,Vxz,Vyz] = rectangle_gravity_matlab('tensor_batch', points, L, B, z_r, G_const, rho_surf, parallel_threshold, chunk_size)

    switch lower(action)
        case 'potential_batch'
            varargout{1} = potential_batch_rectangle(varargin{:});

        case 'acceleration_batch'
            [varargout{1}, varargout{2}, varargout{3}] = acceleration_batch_rectangle(varargin{:});

        case 'tensor_batch'
            [varargout{1}, varargout{2}, varargout{3}, varargout{4}, varargout{5}, varargout{6}] = tensor_batch_rectangle(varargin{:});

        otherwise
            error('Unknown action "%s".', action);
    end
end

% ========================================================================
% Special MATH 
% ========================================================================

function y = sp_log_scalar(arg)
    if arg <= 0
        y = 0.0;
    else
        y = log(arg);
    end
end

function a = sp_arctan_scalar(num, den)
    if (num == 0.0) || (den == 0.0)
        a = 0.0;
    else
        a = atan(num / den);
    end
end

function out = sp_log_vec(x)
% Vector version matching sp_log_scalar exactly
    x = double(x);
    out = zeros(size(x));
    pos = (x > 0.0);
    out(pos) = log(x(pos));
    % x <= 0 -> 0 already
end

function out = sp_arctan_vec(num, den)
% Vector version matching sp_arctan_scalar exactly:
% if num==0 OR den==0 -> 0 else atan(num/den)
    num = double(num); den = double(den);
    out = zeros(size(num));
    mask = (num ~= 0.0) & (den ~= 0.0);
    out(mask) = atan(num(mask) ./ den(mask));
end

% ========================================================================
% POTENTIAL
% ========================================================================

function V = gravitational_potential_rectangle_scalar(x, y, z, L, B, sigma, G, epsz)
% Exact scalar algorithm/corner
    if nargin < 7 || isempty(G);    G = 1.0; end
    if nargin < 8 || isempty(epsz); epsz = 1e-15; end

    u1 = (L - x);
    u2 = -(L + x);
    v1 = (B - y);
    v2 = -(B + y);

    if abs(z) > epsz
        zz = z;
    else
        if z >= 0
            zz = epsz;
        else
            zz = -epsz;
        end
    end

    Vsum = 0.0;
    combos = [ u1, v1, +1.0;
               u1, v2, -1.0;
               u2, v1, -1.0;
               u2, v2, +1.0 ];

    for k = 1:4
        ui = combos(k,1);
        vj = combos(k,2);
        s  = combos(k,3);
        r  = sqrt(ui*ui + vj*vj + zz*zz);

        Vsum = Vsum + s * ( ...
            ui * sp_log_scalar(vj + r) + ...
            vj * sp_log_scalar(ui + r) - ...
            zz * sp_arctan_scalar(ui*vj, zz*r) );
    end

    V = G * sigma * Vsum;
end

function V = rectangle_potential_vectorized(points, L, B, sigma, G, epsz)
% Vectorized
    if nargin < 5 || isempty(G);    G = 1.0; end
    if nargin < 6 || isempty(epsz); epsz = 1e-15; end

    pts = double(points);
    x = pts(:,1); y = pts(:,2); z = pts(:,3);
    N = size(pts,1);

    zz = z;
    mask = abs(z) <= epsz;
    zz(mask & (z >= 0)) = epsz;
    zz(mask & (z <  0)) = -epsz;

    % corners in scalar-loop order
    u = [ L - x,  L - x, -(L + x), -(L + x) ];  % (N,4)
    v = [ B - y, -(B + y),  B - y, -(B + y) ];  % (N,4)
    s = [ +1, -1, -1, +1 ];                     % (1,4)

    r = sqrt(u.^2 + v.^2 + (zz.^2)*ones(1,4));

    term_log = u .* sp_log_vec(v + r) + ...
               v .* sp_log_vec(u + r);

    num = u .* v;
    den = (zz*ones(1,4)) .* r;

    atan_term = sp_arctan_vec(num, den);

    term = term_log - (zz*ones(1,4)) .* atan_term;
    V = sum((ones(N,1)*s) .* term, 2);

    V = G * sigma * V;
end

function V = potential_batch_rectangle(points, L, B, sigma, G, epsz, parallel_threshold, chunk_size)
% Batch + optional parallel chunking (parfor)
    if nargin < 5 || isempty(G); G = 1.0; end
    if nargin < 6 || isempty(epsz); epsz = 1e-15; end
    if nargin < 7 || isempty(parallel_threshold); parallel_threshold = 50000; end
    if nargin < 8 || isempty(chunk_size); chunk_size = 10000; end

    pts = double(points);
    N = size(pts,1);

    if N < parallel_threshold
        V = rectangle_potential_vectorized(pts, L, B, sigma, G, epsz);
        return
    end

    nChunks = ceil(N / chunk_size);
    outCell = cell(nChunks,1);

    parfor c = 1:nChunks
        i1 = (c-1)*chunk_size + 1;
        i2 = min(c*chunk_size, N);
        outCell{c} = rectangle_potential_vectorized(pts(i1:i2,:), L, B, sigma, G, epsz);
    end

    V = vertcat(outCell{:});
end

% ========================================================================
% ACCELERATION
% ========================================================================

function [gx, gy, gz] = rectangle_acceleration_vectorized(points, L, B, sigma, G, epsz)
% gx/gy log_or_inf(A) rule and deterministic first-hit sign (corner order).
    if nargin < 5 || isempty(G); G = 1.0; end
    if nargin < 6 || isempty(epsz); epsz = 0.0; end

    pts = double(points);
    x = pts(:,1); y = pts(:,2); z = pts(:,3);
    N = size(pts,1);

    zz = z;
    mask = abs(z) <= epsz;
    zz(mask & (z >= 0)) = epsz;
    zz(mask & (z <  0)) = -epsz;

    u = [ L - x,  L - x, -(L + x), -(L + x) ];
    v = [ B - y, -(B + y),  B - y, -(B + y) ];
    s = [ +1, -1, -1, +1 ];

    r = sqrt(u.^2 + v.^2 + (zz.^2)*ones(1,4));

    % ---------------- gx: A = v + r, log_or_inf(A) where A==0 -> +Inf
    A_gx = v + r;
    gx_term = sp_log_vec(A_gx);
    gx_term(A_gx == 0.0) = Inf;

    gx = -G * sigma * sum((ones(N,1)*s) .* gx_term, 2);

   
    hit = (A_gx == 0.0);
    has_hit = any(hit,2);
    if any(has_hit)
        first_idx = zeros(N,1);
        for i = 1:N
            if has_hit(i)
                first_idx(i) = find(hit(i,:), 1, 'first');
            end
        end
        corner_sign = s(first_idx(has_hit)).';
        gx(has_hit) = -G * sigma .* corner_sign .* Inf;
    end

    % ---------------- gy: A = u + r, log_or_inf(A) where A==0 -> +Inf
    A_gy = u + r;
    gy_term = sp_log_vec(A_gy);
    gy_term(A_gy == 0.0) = Inf;

    gy = -G * sigma * sum((ones(N,1)*s) .* gy_term, 2);

    hit = (A_gy == 0.0);
    has_hit = any(hit,2);
    if any(has_hit)
        first_idx = zeros(N,1);
        for i = 1:N
            if has_hit(i)
                first_idx(i) = find(hit(i,:), 1, 'first');
            end
        end
        corner_sign = s(first_idx(has_hit)).';
        gy(has_hit) = -G * sigma .* corner_sign .* Inf;
    end

    % ---------------- gz: sp_arctan(ui*vj, zz*rij)
    num = u .* v;
    den = (zz*ones(1,4)) .* r;

    gz_term = sp_arctan_vec(num, den);
    gz = -G * sigma * sum((ones(N,1)*s) .* gz_term, 2);
end

function [gx, gy, gz] = acceleration_batch_rectangle(points, L, B, sigma, G, epsz, parallel_threshold, chunk_size)
% Batch + optional parallel chunking (parfor)
    if nargin < 5 || isempty(G); G = 1.0; end
    if nargin < 6 || isempty(epsz); epsz = 0.0; end
    if nargin < 7 || isempty(parallel_threshold); parallel_threshold = 50000; end
    if nargin < 8 || isempty(chunk_size); chunk_size = 10000; end

    pts = double(points);
    N = size(pts,1);

    if N < parallel_threshold
        [gx,gy,gz] = rectangle_acceleration_vectorized(pts, L, B, sigma, G, epsz);
        return
    end

    nChunks = ceil(N / chunk_size);
    outCell = cell(nChunks,1);

    parfor c = 1:nChunks
        i1 = (c-1)*chunk_size + 1;
        i2 = min(c*chunk_size, N);
        [cgx,cgy,cgz] = rectangle_acceleration_vectorized(pts(i1:i2,:), L, B, sigma, G, epsz);
        outCell{c} = [cgx,cgy,cgz];
    end

    out = vertcat(outCell{:});
    gx = out(:,1); gy = out(:,2); gz = out(:,3);
end

% ========================================================================
% TENSOR
% ========================================================================

function [Vxx,Vyy,Vzz,Vxy,Vxz,Vyz] = rectangle_tensor_vectorized(points, L, B, z_r, G_const, rho_surf)
    pts = double(points);
    x = pts(:,1); y = pts(:,2); z = pts(:,3);
    N = size(pts,1);

    Z = z - z_r;

    X = [ L - x,  L - x, -(L + x), -(L + x) ];
    Y = [ B - y, -(B + y),  B - y, -(B + y) ];
    S = [ +1, -1, -1, +1 ];

    r2 = X.^2 + Y.^2 + (Z.^2)*ones(1,4);
    r  = sqrt(r2);

    Vxx = sum((ones(N,1)*S) .* (X ./ (r .* (Y + r))), 2);
    Vyy = sum((ones(N,1)*S) .* (Y ./ (r .* (X + r))), 2);
    Vxy = sum((ones(N,1)*S) .* (1.0 ./ r), 2);

    Vxz = sum((ones(N,1)*S) .* ((Z*ones(1,4)) ./ (r .* (Y + r))), 2);
    Vyz = sum((ones(N,1)*S) .* ((Z*ones(1,4)) ./ (r .* (X + r))), 2);

    Vzz = zeros(N,1);

    off = (Z ~= 0.0);
    if any(off)
        Xo = X(off,:); Yo = Y(off,:); Zo = (Z(off))*ones(1,4);
        r2o = r2(off,:); ro = r(off,:);

        num = (Xo .* Yo) .* (Xo.^2 + Yo.^2 + 2.0*Zo.^2);
        den = ro .* ((Zo.^2).*r2o + (Xo.^2).*(Yo.^2));

        Vzz(off) = sum((ones(sum(off),1)*S) .* (num ./ den), 2);
    end

    on = ~off;
    if any(on)
        X0 = X(on,:); Y0 = Y(on,:);
        r0 = sqrt(X0.^2 + Y0.^2);
        Vzz(on) = sum((ones(sum(on),1)*S) .* (r0 ./ (X0 .* Y0)), 2); 
    end

    scale = G_const * rho_surf;
    Vxx = scale*Vxx; Vyy = scale*Vyy; Vzz = scale*Vzz;
    Vxy = scale*Vxy; Vxz = scale*Vxz; Vyz = scale*Vyz;
end

function [Vxx,Vyy,Vzz,Vxy,Vxz,Vyz] = tensor_batch_rectangle(points, L, B, z_r, G_const, rho_surf, parallel_threshold, chunk_size)
% Batch + optional parallel chunking (parfor)
    if nargin < 7 || isempty(parallel_threshold); parallel_threshold = 50000; end
    if nargin < 8 || isempty(chunk_size); chunk_size = 10000; end

    pts = double(points);
    N = size(pts,1);

    if N < parallel_threshold
        [Vxx,Vyy,Vzz,Vxy,Vxz,Vyz] = rectangle_tensor_vectorized(pts, L, B, z_r, G_const, rho_surf);
        return
    end

    nChunks = ceil(N / chunk_size);
    outCell = cell(nChunks,1);

    parfor c = 1:nChunks
        i1 = (c-1)*chunk_size + 1;
        i2 = min(c*chunk_size, N);
        [cVxx,cVyy,cVzz,cVxy,cVxz,cVyz] = rectangle_tensor_vectorized(pts(i1:i2,:), L, B, z_r, G_const, rho_surf);
        outCell{c} = [cVxx,cVyy,cVzz,cVxy,cVxz,cVyz];
    end

    out = vertcat(outCell{:});
    Vxx = out(:,1); Vyy = out(:,2); Vzz = out(:,3);
    Vxy = out(:,4); Vxz = out(:,5); Vyz = out(:,6);
end
