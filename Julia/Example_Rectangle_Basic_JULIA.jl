# ======================================================================
# Example_Rectangle_Basic_JULIA.jl
#
# Basic usage example for the RectangleGravity module.
#
# This script demonstrates:
#   • How to load the local module
#   • Input and output data formats
#   • Evaluation of potential, acceleration, and gravity tensor
#   • Assembly of the gravity tensor and Laplace trace check
#
# Intended for:
#   • Verification against MATLAB and Python implementations
#   • Performance benchmarking
#
# Author:
#   Thunendran Periyandy
#
# ======================================================================


include("RectangleGravity.jl")     # loads your module file
using .RectangleGravity            # dot = local module we just included

# ------------------------------------------------------------
# INPUT FORMAT (Rectangle)
# points: N×3 matrix (rows are [x y z])
#
# Parameters:
#   L, B   : half-lengths (rectangle is 2L × 2B in plane z = z_r)
#   sigma  : surface density used for potential + acceleration
#   G      : gravitational constant
#   epsz   : z-regularization for potential/accel 
#
# Tensor uses:
#   z_r      : rectangle plane
#   G_const  : gravitational constant for tensor
#   rho_surf : surface density for tensor scaling (often = sigma)
# ------------------------------------------------------------

# Rectangle geometry / physics (match your Python benchmark)
L = 1.0
B = 1.0
G = 1.0

D   = 1.0
rho = 1.0
sigma = 1.0 * D * rho     # same as Python rectangle example

# Tensor params
z_r      = 0.0
rho_surf = sigma

# Epsilon handling
epsz = 1e-15

# Points: N×3
points = [
     0.0   0.0   1.0
     0.0   0.0  -1.0
     0.37 -0.41  0.83
     1.0   0.0   0.0
]

# ------------------------------------------------------------
# OUTPUT FORMAT
# Potential: Vector{Float64} length N
# Accel:     (gx,gy,gz) each Vector{Float64} length N
# Tensor:    (Vxx,Vyy,Vzz,Vxy,Vxz,Vyz) each Vector{Float64} length N
# ------------------------------------------------------------

# Potential (Vector length N)
V = potential_batch_rectangle(points, L, B, sigma; G=G, epsz=epsz)
println("V (N):")
println(V)

# Acceleration (gx,gy,gz each Vector length N)
gx, gy, gz = acceleration_batch_rectangle(points, L, B, sigma; G=G, epsz=epsz)
g = hcat(gx, gy, gz)   # N×3
println("\ng (N×3):")
println(g)

# Tensor (each component Vector length N)
Vxx, Vyy, Vzz, Vxy, Vxz, Vyz = tensor_batch_rectangle(points, L, B, z_r, G, rho_surf)

# Assemble Γ as (N×3×3) if you want
N = size(points, 1)
Gamma = Array{Float64}(undef, N, 3, 3)
for i in 1:N
    Gamma[i,:,:] = [
        Vxx[i] Vxy[i] Vxz[i];
        Vxy[i] Vyy[i] Vyz[i];
        Vxz[i] Vyz[i] Vzz[i]
    ]
end
println("\nGamma (N×3×3) (assembled symmetric form):")
println(Gamma)

# Laplace trace (Vector length N)
lap = Vxx .+ Vyy .+ Vzz
println("\nLaplace trace (N):")
println(lap)

# ------------------------------------------------------------
# OPTIONAL: dispatcher style (same outputs, different call)
# ------------------------------------------------------------
# V2 = rectangle_gravity("potential_batch", points, L, B, sigma; G=G, epsz=epsz)
# gx2,gy2,gz2 = rectangle_gravity("acceleration_batch", points, L, B, sigma; G=G, epsz=epsz)
# Vxx2,Vyy2,Vzz2,Vxy2,Vxz2,Vyz2 = rectangle_gravity("tensor_batch", points, L, B, z_r, G, rho_surf)
