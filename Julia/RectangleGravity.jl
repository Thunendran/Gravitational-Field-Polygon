# ======================================================================
# RectangleGravity.jl
#
# Analytical gravitational field of a homogeneous planar rectangle
# using closed-form expressions.
#
# This module provides high-performance batch evaluation of:
#   • Gravitational potential
#   • Gravitational acceleration
#   • Gravity-gradient tensor
#
# The implementation is mathematically consistent with the Python and
# MATLAB reference codes and follows identical sign conventions.
#
# Features:
#   • Thread-parallel evaluation using Julia's multithreading
#   • Allocation-free inner kernels for high performance
#   • Machine-precision agreement across languages
#   • Deterministic behavior near edges and vertices
#
# Input format:
#   points :: N×3 matrix (rows are x, y, z)
#
# Output format:
#   Potential      :: Vector{Float64}
#   Acceleration   :: (gx, gy, gz) as vectors
#   Gravity tensor :: Component-wise vectors
#
# Author:
#   Thunendran Periyandy
#
# ======================================================================

module RectangleGravity

export rectangle_gravity,
       potential_batch_rectangle,
       acceleration_batch_rectangle,
       tensor_batch_rectangle

using Base.Threads

# ======================================================================
# SAFE MATH
#   sp_log(arg):    if arg <= 0 => 0 else log(arg)
#   sp_arctan(n,d): if n==0 OR d==0 => 0 else atan(n/d)
# ======================================================================

@inline function sp_log_scalar(arg::Float64)::Float64
    arg <= 0.0 ? 0.0 : log(arg)
end

@inline function sp_arctan_scalar(num::Float64, den::Float64)::Float64
    (num == 0.0 || den == 0.0) ? 0.0 : atan(num / den)
end

@inline function signed_eps(z::Float64, epsz::Float64)::Float64
    (abs(z) > epsz) ? z : ((z >= 0.0) ? epsz : -epsz)
end

# ======================================================================
# POTENTIAL
# ======================================================================

function potential_batch_rectangle(points::AbstractMatrix{<:Real},
                                   L::Float64, B::Float64, sigma::Float64;
                                   G::Float64=1.0, epsz::Float64=1e-15)

    N = size(points, 1)
    V = Vector{Float64}(undef, N)

    @threads for i in 1:N
        @inbounds begin
            x = Float64(points[i,1])
            y = Float64(points[i,2])
            z = Float64(points[i,3])
        end

        zz = signed_eps(z, epsz)

        u1 = L - x
        u2 = -(L + x)
        v1 = B - y
        v2 = -(B + y)

        Vsum = 0.0

        # (u1,v1,+)
        r = sqrt(u1*u1 + v1*v1 + zz*zz)
        Vsum += ( u1*sp_log_scalar(v1 + r) +
                  v1*sp_log_scalar(u1 + r) -
                  zz*sp_arctan_scalar(u1*v1, zz*r) )

        # (u1,v2,-)
        r = sqrt(u1*u1 + v2*v2 + zz*zz)
        Vsum -= ( u1*sp_log_scalar(v2 + r) +
                  v2*sp_log_scalar(u1 + r) -
                  zz*sp_arctan_scalar(u1*v2, zz*r) )

        # (u2,v1,-)
        r = sqrt(u2*u2 + v1*v1 + zz*zz)
        Vsum -= ( u2*sp_log_scalar(v1 + r) +
                  v1*sp_log_scalar(u2 + r) -
                  zz*sp_arctan_scalar(u2*v1, zz*r) )

        # (u2,v2,+)
        r = sqrt(u2*u2 + v2*v2 + zz*zz)
        Vsum += ( u2*sp_log_scalar(v2 + r) +
                  v2*sp_log_scalar(u2 + r) -
                  zz*sp_arctan_scalar(u2*v2, zz*r) )

        V[i] = G * sigma * Vsum
    end

    return V
end

# ======================================================================
# ACCELERATION
# ======================================================================

function acceleration_batch_rectangle(points::AbstractMatrix{<:Real},
                                      L::Float64, B::Float64, sigma::Float64;
                                      G::Float64=1.0, epsz::Float64=0.0)

    N = size(points, 1)
    gx = Vector{Float64}(undef, N)
    gy = Vector{Float64}(undef, N)
    gz = Vector{Float64}(undef, N)

    @threads for i in 1:N
        @inbounds begin
            x = Float64(points[i,1])
            y = Float64(points[i,2])
            z = Float64(points[i,3])
        end

        zz = signed_eps(z, epsz)

        u1 = L - x
        u2 = -(L + x)
        v1 = B - y
        v2 = -(B + y)

        ssum_gx = 0.0
        ssum_gy = 0.0
        ssum_gz = 0.0

        gx_inf = false; gx_inf_sign = 0.0
        gy_inf = false; gy_inf_sign = 0.0

        # Corner 1: (u1,v1,+)
        s = +1.0
        r = sqrt(u1*u1 + v1*v1 + zz*zz)

        A = v1 + r
        if !gx_inf
            if A == 0.0
                gx_inf = true; gx_inf_sign = s
            else
                ssum_gx += s * sp_log_scalar(A)
            end
        end

        A = u1 + r
        if !gy_inf
            if A == 0.0
                gy_inf = true; gy_inf_sign = s
            else
                ssum_gy += s * sp_log_scalar(A)
            end
        end

        ssum_gz += s * sp_arctan_scalar(u1*v1, zz*r)

        # Corner 2: (u1,v2,-)
        s = -1.0
        r = sqrt(u1*u1 + v2*v2 + zz*zz)

        A = v2 + r
        if !gx_inf
            if A == 0.0
                gx_inf = true; gx_inf_sign = s
            else
                ssum_gx += s * sp_log_scalar(A)
            end
        end

        A = u1 + r
        if !gy_inf
            if A == 0.0
                gy_inf = true; gy_inf_sign = s
            else
                ssum_gy += s * sp_log_scalar(A)
            end
        end

        ssum_gz += s * sp_arctan_scalar(u1*v2, zz*r)

        # Corner 3: (u2,v1,-)
        s = -1.0
        r = sqrt(u2*u2 + v1*v1 + zz*zz)

        A = v1 + r
        if !gx_inf
            if A == 0.0
                gx_inf = true; gx_inf_sign = s
            else
                ssum_gx += s * sp_log_scalar(A)
            end
        end

        A = u2 + r
        if !gy_inf
            if A == 0.0
                gy_inf = true; gy_inf_sign = s
            else
                ssum_gy += s * sp_log_scalar(A)
            end
        end

        ssum_gz += s * sp_arctan_scalar(u2*v1, zz*r)

        # Corner 4: (u2,v2,+)
        s = +1.0
        r = sqrt(u2*u2 + v2*v2 + zz*zz)

        A = v2 + r
        if !gx_inf
            if A == 0.0
                gx_inf = true; gx_inf_sign = s
            else
                ssum_gx += s * sp_log_scalar(A)
            end
        end

        A = u2 + r
        if !gy_inf
            if A == 0.0
                gy_inf = true; gy_inf_sign = s
            else
                ssum_gy += s * sp_log_scalar(A)
            end
        end

        ssum_gz += s * sp_arctan_scalar(u2*v2, zz*r)

        gx[i] = gx_inf ? (-G * sigma * gx_inf_sign * Inf) : (-G * sigma * ssum_gx)
        gy[i] = gy_inf ? (-G * sigma * gy_inf_sign * Inf) : (-G * sigma * ssum_gy)
        gz[i] = -G * sigma * ssum_gz
    end

    return gx, gy, gz
end

# ======================================================================
# TENSOR
# Returns: Vxx,Vyy,Vzz,Vxy,Vxz,Vyz
# ======================================================================

function tensor_batch_rectangle(points::AbstractMatrix{<:Real},
                                L::Float64, B::Float64, z_r::Float64,
                                G_const::Float64, rho_surf::Float64)

    N = size(points, 1)
    Vxx = Vector{Float64}(undef, N)
    Vyy = Vector{Float64}(undef, N)
    Vzz = Vector{Float64}(undef, N)
    Vxy = Vector{Float64}(undef, N)
    Vxz = Vector{Float64}(undef, N)
    Vyz = Vector{Float64}(undef, N)

    scale = G_const * rho_surf

    @threads for i in 1:N
        @inbounds begin
            x = Float64(points[i,1])
            y = Float64(points[i,2])
            z = Float64(points[i,3])
        end

        Z = z - z_r

        u1 = L - x
        u2 = -(L + x)
        v1 = B - y
        v2 = -(B + y)

        Vxx_s = 0.0; Vyy_s = 0.0; Vzz_s = 0.0
        Vxy_s = 0.0; Vxz_s = 0.0; Vyz_s = 0.0

        # corner compute (ui,vj) returning the 6 unscaled contributions
        function corner(ui::Float64, vj::Float64)
            r = sqrt(ui*ui + vj*vj + Z*Z)

            vxx = ui / (r * (vj + r))
            vyy = vj / (r * (ui + r))
            vxy = 1.0 / r
            vxz = Z  / (r * (vj + r))
            vyz = Z  / (r * (ui + r))

            vzz = 0.0
            if Z == 0.0
                r0 = sqrt(ui*ui + vj*vj)
                vzz = r0 / (ui * vj)  # SIGN FIX
            else
                r2 = ui*ui + vj*vj + Z*Z
                num = (ui * vj) * (ui*ui + vj*vj + 2.0*Z*Z)  # SIGN FIX
                den = r * ((Z*Z) * r2 + (ui*ui) * (vj*vj))
                vzz = num / den
            end
            return vxx, vyy, vzz, vxy, vxz, vyz
        end

        # order: (u1,v1,+), (u1,v2,-), (u2,v1,-), (u2,v2,+)

        s = +1.0
        a,b,c,d,e,f = corner(u1,v1)
        Vxx_s += s*a; Vyy_s += s*b; Vzz_s += s*c; Vxy_s += s*d; Vxz_s += s*e; Vyz_s += s*f

        s = -1.0
        a,b,c,d,e,f = corner(u1,v2)
        Vxx_s += s*a; Vyy_s += s*b; Vzz_s += s*c; Vxy_s += s*d; Vxz_s += s*e; Vyz_s += s*f

        s = -1.0
        a,b,c,d,e,f = corner(u2,v1)
        Vxx_s += s*a; Vyy_s += s*b; Vzz_s += s*c; Vxy_s += s*d; Vxz_s += s*e; Vyz_s += s*f

        s = +1.0
        a,b,c,d,e,f = corner(u2,v2)
        Vxx_s += s*a; Vyy_s += s*b; Vzz_s += s*c; Vxy_s += s*d; Vxz_s += s*e; Vyz_s += s*f

        Vxx[i] = scale * Vxx_s
        Vyy[i] = scale * Vyy_s
        Vzz[i] = scale * Vzz_s
        Vxy[i] = scale * Vxy_s
        Vxz[i] = scale * Vxz_s
        Vyz[i] = scale * Vyz_s
    end

    return Vxx, Vyy, Vzz, Vxy, Vxz, Vyz
end

# ======================================================================
# DISPATCHER 
# ======================================================================

function rectangle_gravity(action::AbstractString, args...; kwargs...)
    act = lowercase(String(action))
    if act == "potential_batch"
        return potential_batch_rectangle(args...; kwargs...)
    elseif act == "acceleration_batch"
        return acceleration_batch_rectangle(args...; kwargs...)
    elseif act == "tensor_batch"
        return tensor_batch_rectangle(args...; kwargs...)
    else
        error("Unknown action \"$action\"")
    end
end

end # module RectangleGravity
