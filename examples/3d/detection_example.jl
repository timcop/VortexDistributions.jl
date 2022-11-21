using VortexDistributions
using FourierGPE
using JLD2

include("../../src/utils.jl")


## Sim params

# L=(16.,16.,16.);
# N=(64,64,64);
# sim = Sim(L,N);
# @unpack_Sim sim;

# μ = 25.0;
# γ = 0.05;
# tf = 4/γ;
# Nt = 200;
# t = LinRange(0.,tf,Nt);

# x,y,z = X;

data = joinpath(@__DIR__, "../data/box_vorts.jld2")

@load data psi_tubes1 X

psi = psi_tubes1
plot_iso(psi, X, true, true)

# Params 
N = 2
markersize = 400


# This finds all vortex points intersecting the planes in 3 directions
@time vorts_3d = find_vortex_points_3d(psi, X, N) 
plot_iso(psi, X, true, true)
scatterVortsOnIso(vorts_3d)


# This creates an array of sets of connected vortices unordered
@time vorts_class = connect_vortex_points_3d(vorts_3d, X, 0., N, true)
plot_iso(psi, X, true, true)
scatterClassifiedVortices(vorts_class, vorts_3d, X, markersize)

# This orders the vortices
@time v_sort = sort_classified_vorts_3d(vorts_class, vorts_3d, X); 
plot_iso(psi, X, true, true)
periodicPlotting(v_sort, X, 10)
