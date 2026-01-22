# ======================================================================
# TriangleGravity.jl
#
# Analytical gravitational field of a homogeneous triangle
# (uniform surface density σ) using closed-form expressions.
#
# This module provides:
#   • Potential:        potential_batch_triangle(tri, points) -> Vector{Float64} (N)
#   • Acceleration:     acceleration_batch_triangle(tri, points) -> (gx,gy,gz) each Vector{Float64} (N)
#   • Gravity tensor:   tensor_batch_triangle(tri, points) -> 9 component vectors (N)
#                       returned as (G11,G12,G13, G21,G22,G23, G31,G32,G33)
#   • Dispatcher:       triangle_gravity("potential_batch"/"acceleration_batch"/"tensor_batch", tri, points)
#
# ----------------------------------------------------------------------
# INPUT FORMAT
#
#   vertices : 3×3 (rows are v0,v1,v2), Real-valued
#   points   : N×3 (rows are observation points [x y z]), Real-valued
#
# Construction:
#   tri = TriangleLaminaGravitation(vertices; G=1.0, sigma=1.0, eps=0.0)
#
# Parameters:
#   G     : gravitational constant
#   sigma : triangle surface density
#   eps   : small numerical regularization parameter (used in safe inverses and atan2-like logic)
#
# ----------------------------------------------------------------------
# OUTPUT FORMAT
#
# Potential:
#   V :: Vector{Float64} length N
#
# Acceleration:
#   (gx,gy,gz) :: Tuple of 3 vectors, each length N
#   where g = [gx gy gz] is N×3 if assembled with hcat(gx,gy,gz)
#
# Tensor:
#   (G11,G12,G13,G21,G22,G23,G31,G32,G33) :: 9 vectors, each length N
#   corresponding to the unsymmetrized 3×3 tensor Γ per point:
#     [G11 G12 G13;
#      G21 G22 G23;
#      G31 G32 G33]
#
# Laplace trace at each point:
#   lap = G11 .+ G22 .+ G33
#
# ----------------------------------------------------------------------
# IMPLEMENTATION NOTES
#
# • Geometry is precomputed in TriangleLaminaGravitation (unit normal, edge unit tangents, and û vectors).
# • Parallelized over points using Base.Threads.@threads.
#
# ----------------------------------------------------------------------
# Author:
#   Thunendran Periyandy
#
# Purpose:
#   Cross-language (Python / MATLAB / Julia) validation and high-performance
#   evaluation of analytical gravitational fields for Polygon.
#
# ======================================================================

module TriangleGravity

export TriangleLaminaGravitation,
       potential_batch_triangle,
       acceleration_batch_triangle,
       tensor_batch_triangle,
       triangle_gravity

using Base.Threads

# ======================================================================
# Special MATH
# ======================================================================

@inline function sp_log_scalar(x::Float64)::Float64
    x <= 0.0 ? 0.0 : log(x)
end

"""
atan2_like(y,x,eps): replicate sp_atan2_like_vec WITHOUT atan2.
Returns angle in (-pi, pi], with eps handling for x≈0.
"""
@inline function atan2_like_scalar(y::Float64, x::Float64, eps::Float64)::Float64
    if abs(x) > eps
        base = atan(y / x)
        if x < 0.0
            return (y >= 0.0) ? (base + π) : (base - π)
        else
            return base
        end
    else
        if y > eps
            return 0.5 * π
        elseif y < -eps
            return -0.5 * π
        else
            return 0.0
        end
    end
end

@inline function safe_inv(r::Float64, eps::Float64)::Float64
    r > eps ? 1.0 / r : 0.0
end

# ======================================================================
# SMALL VEC3 HELPERS
# ======================================================================

@inline function dot3(ax::Float64,ay::Float64,az::Float64,
                      bx::Float64,by::Float64,bz::Float64)::Float64
    return ax*bx + ay*by + az*bz
end

@inline function cross3(ax::Float64,ay::Float64,az::Float64,
                        bx::Float64,by::Float64,bz::Float64)
    return (ay*bz - az*by,
            az*bx - ax*bz,
            ax*by - ay*bx)
end

@inline function norm3(ax::Float64,ay::Float64,az::Float64)::Float64
    return sqrt(ax*ax + ay*ay + az*az)
end

# ======================================================================
# TRIANGLE OBJECT (PRECOMPUTES GEOMETRY)
# ======================================================================

struct TriangleLaminaGravitation
    G::Float64
    sigma::Float64
    eps::Float64

    # vertices
    v0x::Float64; v0y::Float64; v0z::Float64
    v1x::Float64; v1y::Float64; v1z::Float64
    v2x::Float64; v2y::Float64; v2z::Float64

    # unit normal
    nx::Float64; ny::Float64; nz::Float64

    # edges endpoints ea, eb (3 edges) packed as scalars
    ea::NTuple{9,Float64}   # (ea0x,ea0y,ea0z, ea1x,... ea2z)
    eb::NTuple{9,Float64}

    # edge lengths
    L0::Float64; L1::Float64; L2::Float64

    # t_hat for each edge (unit tangent)
    t0::NTuple{3,Float64}
    t1::NTuple{3,Float64}
    t2::NTuple{3,Float64}

    # u_hat for each edge = cross(t_hat, n_hat)
    u0::NTuple{3,Float64}
    u1::NTuple{3,Float64}
    u2::NTuple{3,Float64}
end

function TriangleLaminaGravitation(vertices::AbstractMatrix{<:Real};
                                   G::Real=1.0, sigma::Real=1.0, eps::Real=0.0)

    v = Float64.(vertices)
    size(v,1)==3 && size(v,2)==3 || error("vertices must be 3×3: [v0; v1; v2]")

    v0x,v0y,v0z = v[1,1],v[1,2],v[1,3]
    v1x,v1y,v1z = v[2,1],v[2,2],v[2,3]
    v2x,v2y,v2z = v[3,1],v[3,2],v[3,3]

    # normal
    e01x,e01y,e01z = v1x-v0x, v1y-v0y, v1z-v0z
    e02x,e02y,e02z = v2x-v0x, v2y-v0y, v2z-v0z
    nx,ny,nz = cross3(e01x,e01y,e01z, e02x,e02y,e02z)
    nn = norm3(nx,ny,nz)
    (isfinite(nn) && nn > eps) || error("Degenerate triangle (area ~0) or invalid normal.")
    nx/=nn; ny/=nn; nz/=nn

    # directed edges: (0->1),(1->2),(2->0)
    ea = (v0x,v0y,v0z,  v1x,v1y,v1z,  v2x,v2y,v2z)
    eb = (v1x,v1y,v1z,  v2x,v2y,v2z,  v0x,v0y,v0z)

    # edge vectors and lengths
    e0x,e0y,e0z = eb[1]-ea[1], eb[2]-ea[2], eb[3]-ea[3]
    e1x,e1y,e1z = eb[4]-ea[4], eb[5]-ea[5], eb[6]-ea[6]
    e2x,e2y,e2z = eb[7]-ea[7], eb[8]-ea[8], eb[9]-ea[9]

    L0 = norm3(e0x,e0y,e0z)
    L1 = norm3(e1x,e1y,e1z)
    L2 = norm3(e2x,e2y,e2z)
    (L0>eps && L1>eps && L2>eps) || error("Degenerate edge length encountered.")

    # t_hat
    t0 = (e0x/L0, e0y/L0, e0z/L0)
    t1 = (e1x/L1, e1y/L1, e1z/L1)
    t2 = (e2x/L2, e2y/L2, e2z/L2)

    # u_hat = cross(t_hat, n_hat)
    u0 = cross3(t0[1],t0[2],t0[3], nx,ny,nz)
    u1 = cross3(t1[1],t1[2],t1[3], nx,ny,nz)
    u2 = cross3(t2[1],t2[2],t2[3], nx,ny,nz)

    return TriangleLaminaGravitation(Float64(G), Float64(sigma), Float64(eps),
        v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z,
        nx,ny,nz,
        ea, eb,
        L0,L1,L2,
        t0,t1,t2,
        u0,u1,u2
    )
end

# ======================================================================
# LOG TERM
# ======================================================================

@inline function log_term_scalar(ra::Float64, rb::Float64, L::Float64, eps::Float64)::Float64
    s = ra + rb
    den = s - L
    if (L > eps) && (den > eps)
        ratio = (s + L) / den
        if ratio > 0.0
            return sp_log_scalar(ratio)
        end
    end
    return 0.0
end

# ======================================================================
# SOLID ANGLE + GRAD
# Returns Omega::Float64 and gradOmega as (gx,gy,gz)
# ======================================================================

function solid_angle_and_grad_scalar(Px::Float64,Py::Float64,Pz::Float64,
                                     v0x::Float64,v0y::Float64,v0z::Float64,
                                     v1x::Float64,v1y::Float64,v1z::Float64,
                                     v2x::Float64,v2y::Float64,v2z::Float64,
                                     eps::Float64)

    # r0 = v0 - P, etc
    r0x,r0y,r0z = v0x-Px, v0y-Py, v0z-Pz
    r1x,r1y,r1z = v1x-Px, v1y-Py, v1z-Pz
    r2x,r2y,r2z = v2x-Px, v2y-Py, v2z-Pz

    r0n = norm3(r0x,r0y,r0z)
    r1n = norm3(r1x,r1y,r1z)
    r2n = norm3(r2x,r2y,r2z)

    # crosses
    c12x,c12y,c12z = cross3(r1x,r1y,r1z, r2x,r2y,r2z)
    c20x,c20y,c20z = cross3(r2x,r2y,r2z, r0x,r0y,r0z)
    c01x,c01y,c01z = cross3(r0x,r0y,r0z, r1x,r1y,r1z)

    # T = dot(c12, r0)
    T = dot3(c12x,c12y,c12z, r0x,r0y,r0z)

    d01 = dot3(r0x,r0y,r0z, r1x,r1y,r1z)
    d12 = dot3(r1x,r1y,r1z, r2x,r2y,r2z)
    d20 = dot3(r2x,r2y,r2z, r0x,r0y,r0z)

    D = (r0n*r1n*r2n + d01*r2n + d12*r0n + d20*r1n)

    Omega = 2.0 * atan2_like_scalar(T, D, eps)

    # gradT = -(c12 + c20 + c01)
    gradTx = -(c12x + c20x + c01x)
    gradTy = -(c12y + c20y + c01y)
    gradTz = -(c12z + c20z + c01z)

    inv_r0 = safe_inv(r0n, eps)
    inv_r1 = safe_inv(r1n, eps)
    inv_r2 = safe_inv(r2n, eps)

    # grad_r0n = -r0 / |r0|
    grad_r0x, grad_r0y, grad_r0z = -r0x*inv_r0, -r0y*inv_r0, -r0z*inv_r0
    grad_r1x, grad_r1y, grad_r1z = -r1x*inv_r1, -r1y*inv_r1, -r1z*inv_r1
    grad_r2x, grad_r2y, grad_r2z = -r2x*inv_r2, -r2y*inv_r2, -r2z*inv_r2

    # grad_d01 = -(r0 + r1)
    grad_d01x, grad_d01y, grad_d01z = -(r0x+r1x), -(r0y+r1y), -(r0z+r1z)
    grad_d12x, grad_d12y, grad_d12z = -(r1x+r2x), -(r1y+r2y), -(r1z+r2z)
    grad_d20x, grad_d20y, grad_d20z = -(r2x+r0x), -(r2y+r0y), -(r2z+r0z)

    # gradD pieces
    # gradD1 = grad_r0n*(r1n*r2n) + grad_r1n*(r0n*r2n) + grad_r2n*(r0n*r1n)
    t01 = r1n*r2n
    t02 = r0n*r2n
    t03 = r0n*r1n
    gradD1x = grad_r0x*t01 + grad_r1x*t02 + grad_r2x*t03
    gradD1y = grad_r0y*t01 + grad_r1y*t02 + grad_r2y*t03
    gradD1z = grad_r0z*t01 + grad_r1z*t02 + grad_r2z*t03

    # gradD2 = grad_d01*r2n + grad_r2n*d01
    gradD2x = grad_d01x*r2n + grad_r2x*d01
    gradD2y = grad_d01y*r2n + grad_r2y*d01
    gradD2z = grad_d01z*r2n + grad_r2z*d01

    # gradD3 = grad_d12*r0n + grad_r0n*d12
    gradD3x = grad_d12x*r0n + grad_r0x*d12
    gradD3y = grad_d12y*r0n + grad_r0y*d12
    gradD3z = grad_d12z*r0n + grad_r0z*d12

    # gradD4 = grad_d20*r1n + grad_r1n*d20
    gradD4x = grad_d20x*r1n + grad_r1x*d20
    gradD4y = grad_d20y*r1n + grad_r1y*d20
    gradD4z = grad_d20z*r1n + grad_r1z*d20

    gradDx = gradD1x + gradD2x + gradD3x + gradD4x
    gradDy = gradD1y + gradD2y + gradD3y + gradD4y
    gradDz = gradD1z + gradD2z + gradD3z + gradD4z

    denom = T*T + D*D
    if denom > eps && isfinite(denom)
        # gradOmega = 2 * (D*gradT - T*gradD) / (T^2 + D^2)
        gx = 2.0 * (D*gradTx - T*gradDx) / denom
        gy = 2.0 * (D*gradTy - T*gradDy) / denom
        gz = 2.0 * (D*gradTz - T*gradDz) / denom
        if !isfinite(Omega); Omega = 0.0; end
        if !(isfinite(gx) && isfinite(gy) && isfinite(gz))
            return Omega, 0.0, 0.0, 0.0
        end
        return Omega, gx, gy, gz
    else
        if !isfinite(Omega); Omega = 0.0; end
        return Omega, 0.0, 0.0, 0.0
    end
end

# ======================================================================
# BATCH: POTENTIAL / ACCELERATION / TENSOR
# ======================================================================

function potential_batch_triangle(tri::TriangleLaminaGravitation,
                                  points::AbstractMatrix{<:Real})
    N = size(points, 1)
    out = Vector{Float64}(undef, N)

    @threads for i in 1:N
        @inbounds begin
            Px = Float64(points[i,1])
            Py = Float64(points[i,2])
            Pz = Float64(points[i,3])
        end

        # z = dot(v0 - P, n_hat)
        z = dot3(tri.v0x-Px, tri.v0y-Py, tri.v0z-Pz, tri.nx, tri.ny, tri.nz)

        # Omega
        Omega, _, _, _ = solid_angle_and_grad_scalar(
            Px,Py,Pz,
            tri.v0x,tri.v0y,tri.v0z,
            tri.v1x,tri.v1y,tri.v1z,
            tri.v2x,tri.v2y,tri.v2z,
            tri.eps
        )

        # edge vectors & distances
        # edge 0: ea=v0, eb=v1
        ra0x,ra0y,ra0z = tri.ea[1]-Px, tri.ea[2]-Py, tri.ea[3]-Pz
        rb0x,rb0y,rb0z = tri.eb[1]-Px, tri.eb[2]-Py, tri.eb[3]-Pz
        ra0 = norm3(ra0x,ra0y,ra0z)
        rb0 = norm3(rb0x,rb0y,rb0z)

        ra1x,ra1y,ra1z = tri.ea[4]-Px, tri.ea[5]-Py, tri.ea[6]-Pz
        rb1x,rb1y,rb1z = tri.eb[4]-Px, tri.eb[5]-Py, tri.eb[6]-Pz
        ra1 = norm3(ra1x,ra1y,ra1z)
        rb1 = norm3(rb1x,rb1y,rb1z)

        ra2x,ra2y,ra2z = tri.ea[7]-Px, tri.ea[8]-Py, tri.ea[9]-Pz
        rb2x,rb2y,rb2z = tri.eb[7]-Px, tri.eb[8]-Py, tri.eb[9]-Pz
        ra2 = norm3(ra2x,ra2y,ra2z)
        rb2 = norm3(rb2x,rb2y,rb2z)

        # p = dot(ra_vec, u_hat)
        p0 = dot3(ra0x,ra0y,ra0z, tri.u0[1],tri.u0[2],tri.u0[3])
        p1 = dot3(ra1x,ra1y,ra1z, tri.u1[1],tri.u1[2],tri.u1[3])
        p2 = dot3(ra2x,ra2y,ra2z, tri.u2[1],tri.u2[2],tri.u2[3])

        logE0 = log_term_scalar(ra0, rb0, tri.L0, tri.eps)
        logE1 = log_term_scalar(ra1, rb1, tri.L1, tri.eps)
        logE2 = log_term_scalar(ra2, rb2, tri.L2, tri.eps)

        edge_sum = p0*logE0 + p1*logE1 + p2*logE2

        out[i] = tri.G * tri.sigma * (edge_sum - z * Omega)
    end

    return out
end

function acceleration_batch_triangle(tri::TriangleLaminaGravitation,
                                     points::AbstractMatrix{<:Real})
    N = size(points, 1)
    gx = Vector{Float64}(undef, N)
    gy = Vector{Float64}(undef, N)
    gz = Vector{Float64}(undef, N)

    @threads for i in 1:N
        @inbounds begin
            Px = Float64(points[i,1])
            Py = Float64(points[i,2])
            Pz = Float64(points[i,3])
        end

        Omega, _, _, _ = solid_angle_and_grad_scalar(
            Px,Py,Pz,
            tri.v0x,tri.v0y,tri.v0z,
            tri.v1x,tri.v1y,tri.v1z,
            tri.v2x,tri.v2y,tri.v2z,
            tri.eps
        )

        # edges distances
        ra0 = norm3(tri.ea[1]-Px, tri.ea[2]-Py, tri.ea[3]-Pz)
        rb0 = norm3(tri.eb[1]-Px, tri.eb[2]-Py, tri.eb[3]-Pz)
        ra1 = norm3(tri.ea[4]-Px, tri.ea[5]-Py, tri.ea[6]-Pz)
        rb1 = norm3(tri.eb[4]-Px, tri.eb[5]-Py, tri.eb[6]-Pz)
        ra2 = norm3(tri.ea[7]-Px, tri.ea[8]-Py, tri.ea[9]-Pz)
        rb2 = norm3(tri.eb[7]-Px, tri.eb[8]-Py, tri.eb[9]-Pz)

        logE0 = log_term_scalar(ra0, rb0, tri.L0, tri.eps)
        logE1 = log_term_scalar(ra1, rb1, tri.L1, tri.eps)
        logE2 = log_term_scalar(ra2, rb2, tri.L2, tri.eps)

        # edge_vec = Σ logE * u_hat
        ex = logE0*tri.u0[1] + logE1*tri.u1[1] + logE2*tri.u2[1]
        ey = logE0*tri.u0[2] + logE1*tri.u1[2] + logE2*tri.u2[2]
        ez = logE0*tri.u0[3] + logE1*tri.u1[3] + logE2*tri.u2[3]

        # face_vec = -Omega * n_hat
        fx = -Omega * tri.nx
        fy = -Omega * tri.ny
        fz = -Omega * tri.nz

        gx[i] = tri.G * tri.sigma * (ex + fx)
        gy[i] = tri.G * tri.sigma * (ey + fy)
        gz[i] = tri.G * tri.sigma * (ez + fz)
    end

    return gx, gy, gz
end

function tensor_batch_triangle(tri::TriangleLaminaGravitation,
                               points::AbstractMatrix{<:Real})
    N = size(points, 1)

    # store 9 components row-major (g_i w.r.t x_j)
    G11 = Vector{Float64}(undef, N); G12 = Vector{Float64}(undef, N); G13 = Vector{Float64}(undef, N)
    G21 = Vector{Float64}(undef, N); G22 = Vector{Float64}(undef, N); G23 = Vector{Float64}(undef, N)
    G31 = Vector{Float64}(undef, N); G32 = Vector{Float64}(undef, N); G33 = Vector{Float64}(undef, N)

    @threads for i in 1:N
        @inbounds begin
            Px = Float64(points[i,1])
            Py = Float64(points[i,2])
            Pz = Float64(points[i,3])
        end

        Omega, gΩx, gΩy, gΩz = solid_angle_and_grad_scalar(
            Px,Py,Pz,
            tri.v0x,tri.v0y,tri.v0z,
            tri.v1x,tri.v1y,tri.v1z,
            tri.v2x,tri.v2y,tri.v2z,
            tri.eps
        )

        # edge endpoint vectors
        ra0x,ra0y,ra0z = tri.ea[1]-Px, tri.ea[2]-Py, tri.ea[3]-Pz
        rb0x,rb0y,rb0z = tri.eb[1]-Px, tri.eb[2]-Py, tri.eb[3]-Pz
        ra1x,ra1y,ra1z = tri.ea[4]-Px, tri.ea[5]-Py, tri.ea[6]-Pz
        rb1x,rb1y,rb1z = tri.eb[4]-Px, tri.eb[5]-Py, tri.eb[6]-Pz
        ra2x,ra2y,ra2z = tri.ea[7]-Px, tri.ea[8]-Py, tri.ea[9]-Pz
        rb2x,rb2y,rb2z = tri.eb[7]-Px, tri.eb[8]-Py, tri.eb[9]-Pz

        ra0 = norm3(ra0x,ra0y,ra0z); rb0 = norm3(rb0x,rb0y,rb0z)
        ra1 = norm3(ra1x,ra1y,ra1z); rb1 = norm3(rb1x,rb1y,rb1z)
        ra2 = norm3(ra2x,ra2y,ra2z); rb2 = norm3(rb2x,rb2y,rb2z)

        s0 = ra0 + rb0
        s1 = ra1 + rb1
        s2 = ra2 + rb2

        # factor = 2L/(s^2 - L^2) ;
        function gradlog_edge(ra::Float64, rb::Float64, s::Float64, L::Float64,
                              ra⃗x::Float64,ra⃗y::Float64,ra⃗z::Float64,
                              rb⃗x::Float64,rb⃗y::Float64,rb⃗z::Float64,
                              eps::Float64)

            denom2 = s*s - L*L
            if (L > eps) && (ra > eps) && (rb > eps) && (abs(denom2) > eps)
                factor = (2.0 * L) / denom2
                invra = 1.0 / ra
                invrb = 1.0 / rb
                # dirsum = (A-P)/|A-P| + (B-P)/|B-P|
                dx = ra⃗x*invra + rb⃗x*invrb
                dy = ra⃗y*invra + rb⃗y*invrb
                dz = ra⃗z*invra + rb⃗z*invrb
                # gradLog = factor * dirsum  (SIGN kept as your python)
                return factor*dx, factor*dy, factor*dz
            else
                return 0.0, 0.0, 0.0
            end
        end

        gl0x,gl0y,gl0z = gradlog_edge(ra0,rb0,s0,tri.L0, ra0x,ra0y,ra0z, rb0x,rb0y,rb0z, tri.eps)
        gl1x,gl1y,gl1z = gradlog_edge(ra1,rb1,s1,tri.L1, ra1x,ra1y,ra1z, rb1x,rb1y,rb1z, tri.eps)
        gl2x,gl2y,gl2z = gradlog_edge(ra2,rb2,s2,tri.L2, ra2x,ra2y,ra2z, rb2x,rb2y,rb2z, tri.eps)

        # edge_tensor_sum = Σ u_e ⊗ gradLog_e
        # u ⊗ g means (i,j) = u_i * g_j
        ex11 = tri.u0[1]*gl0x + tri.u1[1]*gl1x + tri.u2[1]*gl2x
        ex12 = tri.u0[1]*gl0y + tri.u1[1]*gl1y + tri.u2[1]*gl2y
        ex13 = tri.u0[1]*gl0z + tri.u1[1]*gl1z + tri.u2[1]*gl2z

        ex21 = tri.u0[2]*gl0x + tri.u1[2]*gl1x + tri.u2[2]*gl2x
        ex22 = tri.u0[2]*gl0y + tri.u1[2]*gl1y + tri.u2[2]*gl2y
        ex23 = tri.u0[2]*gl0z + tri.u1[2]*gl1z + tri.u2[2]*gl2z

        ex31 = tri.u0[3]*gl0x + tri.u1[3]*gl1x + tri.u2[3]*gl2x
        ex32 = tri.u0[3]*gl0y + tri.u1[3]*gl1y + tri.u2[3]*gl2y
        ex33 = tri.u0[3]*gl0z + tri.u1[3]*gl1z + tri.u2[3]*gl2z

        # face_tensor_sum = n ⊗ gradOmega
        fx11 = tri.nx*gΩx; fx12 = tri.nx*gΩy; fx13 = tri.nx*gΩz
        fx21 = tri.ny*gΩx; fx22 = tri.ny*gΩy; fx23 = tri.ny*gΩz
        fx31 = tri.nz*gΩx; fx32 = tri.nz*gΩy; fx33 = tri.nz*gΩz

        sconst = tri.G * tri.sigma

        G11[i] = sconst * (ex11 - fx11)
        G12[i] = sconst * (ex12 - fx12)
        G13[i] = sconst * (ex13 - fx13)

        G21[i] = sconst * (ex21 - fx21)
        G22[i] = sconst * (ex22 - fx22)
        G23[i] = sconst * (ex23 - fx23)

        G31[i] = sconst * (ex31 - fx31)
        G32[i] = sconst * (ex32 - fx32)
        G33[i] = sconst * (ex33 - fx33)

        # match python: if non-finite, zero it
        if !(isfinite(G11[i])); G11[i]=0.0; end
        if !(isfinite(G12[i])); G12[i]=0.0; end
        if !(isfinite(G13[i])); G13[i]=0.0; end
        if !(isfinite(G21[i])); G21[i]=0.0; end
        if !(isfinite(G22[i])); G22[i]=0.0; end
        if !(isfinite(G23[i])); G23[i]=0.0; end
        if !(isfinite(G31[i])); G31[i]=0.0; end
        if !(isfinite(G32[i])); G32[i]=0.0; end
        if !(isfinite(G33[i])); G33[i]=0.0; end
    end

    return G11,G12,G13, G21,G22,G23, G31,G32,G33
end

# ======================================================================
# DISPATCHER
# ======================================================================

function triangle_gravity(action::AbstractString, tri::TriangleLaminaGravitation, args...; kwargs...)
    act = lowercase(String(action))
    if act == "potential_batch"
        return potential_batch_triangle(tri, args...)
    elseif act == "acceleration_batch"
        return acceleration_batch_triangle(tri, args...)
    elseif act == "tensor_batch"
        return tensor_batch_triangle(tri, args...)
    else
        error("Unknown action \"$action\"")
    end
end

end # module TriangleGravity
