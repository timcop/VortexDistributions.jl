using VortexDistributions
using JLD2


data = joinpath(@__DIR__, "../../test/3d/box_vorts.jld2")

@load data psi_tubes1 X

include("../../src/utils_plots.jl")


psi = psi_tubes1
plot_iso(psi, X, false)

# Params 
N = 2



# This finds all vortex points intersecting the planes in 3 directions
@time vorts_3d = find_vortex_points_3d(psi, X, N) 
plot_iso(psi, X, false)
scatterVortsOnIso(vorts_3d)


# This creates an array of sets of connected vortices unordered
@time vorts_class = connect_vortex_points_3d(vorts_3d, X, 0., N, true)
plot_iso(psi, X, false)

scatterClassifiedVortices(vorts_class, vorts_3d, X)

# This orders the vortices
@time v_sort = sort_classified_vorts_3d(vorts_class, vorts_3d, X); 
plot_iso(psi, X, false)
periodicPlotting(v_sort, X, 10)
