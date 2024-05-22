using VortexDistributions
using JLD2


data = joinpath(@__DIR__, "quench_slab_jld2s/quenchslab_t60.jld2")

@load data psi X

include("../../src/utils_plots.jl")


psi = psi
plot_iso(psi, X, false)

# Params 
N = 2



# This finds all vortex points intersecting the planes in 3 directions
@time vorts_3d = find_vortex_points_3d(psi, X, N) 
plot_iso(psi, X, false)
scatterVortsOnIso(vorts_3d)

function filterVortBounds(vorts, xb1, xb2, yb1, yb2, zb1, zb2)
    filt_vorts = []
    for i in eachindex(vorts)
        v = vorts[i]
        if ((v[1] >= xb1 && v[1] <= xb2) && (v[2] >= yb1 && v[2] <= yb2) && (v[3] >= zb1 && v[3] <= zb2))
            push!(filt_vorts, v)
        end
    end
    return filt_vorts
end

X

vorts_3d


# This creates an array of sets of connected vortices unordered
@time vorts_class = connect_vortex_points_3d(vorts_3d, X, 0., N, true)
plot_iso(psi, X, false)

scatterClassifiedVortices(vorts_class, vorts_3d, X)

# This orders the vortices
@time v_sort = sort_classified_vorts_3d(vorts_class, vorts_3d, X); 
plot_iso(psi, X, false)
periodicPlotting(v_sort, X, 10)
