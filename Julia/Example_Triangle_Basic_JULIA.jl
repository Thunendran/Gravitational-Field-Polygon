# ======================================================================
# Example_Triangle_Basic_JULIA.jl
#
# Minimal usage example for TriangleGravity.jl.
# Demonstrates how to:
#   1) include and load the local TriangleGravity module,
#   2) construct a TriangleLaminaGravitation object from vertices,
#   3) evaluate potential, acceleration, and gravity tensor at a set of points,
#   4) assemble the 9 tensor component vectors into an (N×3×3) array,
#   5) compute the Laplace trace (Γ11 + Γ22 + Γ33).
#
# ----------------------------------------------------------------------
# INPUT FORMAT
#   vertices : 3×3 matrix (Float64), rows are vertices [v0; v1; v2]
#   points   : N×3 matrix (Float64), rows are observation points [x y z]
#
# OUTPUT FORMAT
#   V                   : Vector{Float64} length N
#   (gx,gy,gz)          : three Vector{Float64} (each length N)
#   (G11..G33)          : nine Vector{Float64} (each length N), row-major order
#   Gamma (assembled)   : Array{Float64} size (N,3,3)
#   lap                 : Vector{Float64} length N
#
# ----------------------------------------------------------------------
# Notes:
# • This script assumes TriangleGravity.jl is in the same folder.
# • For readable printing, you can use:
#     using Printf
#     @printf("%.15e\n", value)
#
# Author:
#   Thunendran Periyandy
#
# ======================================================================


# Example_Triangle_Basic_JULIA.jl

include("TriangleGravity.jl")     # loads your module file
using .TriangleGravity            # the dot means "local module we just included"

# Triangle vertices: 3×3 (rows are v0,v1,v2)
vertices = [
    0.0  0.0  0.0
    1.0  0.0  0.0
    0.0  1.0  0.0
]

G     = 1.0
sigma = 1.0
eps   = 1e-15

tri = TriangleLaminaGravitation(vertices; G=G, sigma=sigma, eps=eps)

# Points: N×3
points = [
    0.2  0.2  0.0
    0.2  0.2  1.0
    0.2  0.2 -1.0
    0.5  0.0  0.0
]

# Potential (Vector length N)
V = potential_batch_triangle(tri, points)
println("V (N):"); println(V)

# Acceleration (gx,gy,gz each Vector length N)
gx, gy, gz = acceleration_batch_triangle(tri, points)
g = hcat(gx, gy, gz)
println("g (N×3):"); println(g)

# Tensor (each component Vector length N)
G11,G12,G13, G21,G22,G23, G31,G32,G33 = tensor_batch_triangle(tri, points)

# Assemble Γ as (N×3×3) if you want
N = size(points,1)
Gamma = Array{Float64}(undef, N, 3, 3)
for i in 1:N
    Gamma[i,:,:] = [
        G11[i] G12[i] G13[i];
        G21[i] G22[i] G23[i];
        G31[i] G32[i] G33[i]
    ]
end
println("Gamma (N×3×3):"); println(Gamma)

# Laplace trace (Vector length N)
lap = G11 .+ G22 .+ G33
println("Laplace trace (N):"); println(lap)
