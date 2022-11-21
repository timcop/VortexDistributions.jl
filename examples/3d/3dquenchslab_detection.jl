using FourierGPE, VortexDistributions, JLD2
using Interpolations 
using JLD2
using Parameters
using SpecialFunctions
using LinearAlgebra
using ToeplitzMatrices
using SparseArrays
using FFTW
using FileIO
using ProgressMeter
using LightGraphs
using SimpleWeightedGraphs

# 3d deps
# using GLMakie
using ScikitLearn
using NearestNeighbors
using Distances
using FLoops

include("../../src/utils_plots.jl")


@load "examples/3d/quench_slab_jld2s/quenchslab_t200.jld2" psi X


plot_iso(psi, X)


function find_vortex_points_3d_harmonic(
    psi :: Array{ComplexF64, 3}, 
    # X :: Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}, 
    X, 

    N :: Int = 1
    ) :: Vector{Vector{Float64}}
    # TODO: Add periodic checks 
    # @assert N <= 16
    # @assert N >= 1
    # print("inner: " + N)

    
    x = X[1]; y = X[2]; z = X[3];
    dx = x[2]-x[1]; dy = y[2]-y[1]; dz = z[2]-z[1];


    # xlim1 = 1; xlim2 = length(x);
    # ylim1 = 1; ylim2 = length(y);
    # zlim1 = 1; zlim2 = length(z);

    # x = x[xlim1:xlim2]; y = y[ylim1:ylim2]; z = z[zlim1:zlim2]

    # @assert size(psi)[1] == length(x)
    # @assert size(psi)[2] == length(y)
    # @assert size(psi)[3] == length(z)

    x_itp = interpolate(x, BSpline(Linear()));
    y_itp = interpolate(y, BSpline(Linear()));
    z_itp = interpolate(z, BSpline(Linear()));

    x_etp = extrapolate(x_itp, Line())
    y_etp = extrapolate(y_itp, Line())
    z_etp = extrapolate(z_itp, Line())

    # psi_itp = interpolate(psi, BSpline(Quadratic((Periodic(OnCell()), Periodic(OnCell()), Line(OnCell())))))
    psi_itp = interpolate(psi, BSpline(Cubic(Line(OnGrid()))))
    psi_etp = extrapolate(psi_itp, (Periodic(), Periodic(), Line()))
    # psi_etp = extrapolate(psi_itp, Periodic())

    x_range = LinRange(1,length(x),N*(length(x)))
    y_range = LinRange(1,length(y),N*(length(y)))

    z_range = LinRange(1,length(z), N*(length(z)))

    # x = LinRange(x[1]-2*dx, x[end]+2*dx, length(x)+4);
    # y = LinRange(y[1]-2*dy, y[end]+2*dy, length(y)+4);
    x = LinRange(x[1], x[end], length(x));
    y = LinRange(y[1], y[end], length(y));
    z = LinRange(z[1], z[end], length(z));

    # x_range = LinRange(1,length(x),N*(length(x)))
    # y_range = LinRange(1,length(y),N*(length(y)))
    # z_range = LinRange(1,length(z), N*(length(z)))

    # x = LinRange(x[1], x[end], length(x));
    # y = LinRange(y[1], y[end], length(y));
    # z = LinRange(z[1], z[end], length(z));

    ## loop vectorisation, run in parallel 
    vorts3d = []
    vorts_xslice = []
    vorts_yslice = []
    vorts_zslice = []

    results_x = [[] for _ in 1:Threads.nthreads()]
    results_y = [[] for _ in 1:Threads.nthreads()]
    results_z = [[] for _ in 1:Threads.nthreads()]

    let z = z, y=y
        @floop for xidx in x_range
            vorts_x = vortex_array(findvortices(Torus(psi_etp(xidx, y_range[1]:y_range[end], z_range[1]:z_range[end]), y, z)))
            for vidx_x in 1:size(vorts_x)[1]
                v_x = vorts_x[vidx_x, :]
                vx_x = [x_etp(xidx), v_x[1], v_x[2], v_x[3]]
                push!(results_x[Threads.threadid()], vx_x)
            end
        end
    end

    let x=x, z=z
        @floop for yidx in y_range
            vorts_y = vortex_array(findvortices(Torus(psi_etp(x_range[1]:x_range[end], yidx, z_range[1]:z_range[end]), x, z)))
            for vidx_y in 1:size(vorts_y)[1]
                v_y = vorts_y[vidx_y, :]
                vy_y = [v_y[1], y_etp(yidx), v_y[2], v_y[3]]
                push!(results_y[Threads.threadid()], vy_y)
            end
        end
    end

    let x=x, y=y
        @floop for zidx in reverse(z_range)
            vorts_z = vortex_array(findvortices(Torus(psi_etp(x_range[1]:x_range[end], y_range[1]:y_range[end], zidx), x, y)))
            for vidx_z in 1:size(vorts_z)[1]
                v_z = vorts_z[vidx_z, :]
                vz_z = [v_z[1], v_z[2], z_etp(zidx), v_z[3]]
                push!(results_z[Threads.threadid()], vz_z)
            end
        end
    end

    vorts_xslice = reduce(vcat, results_x)
    vorts_yslice = reduce(vcat, results_y)
    vorts_zslice = reduce(vcat, results_z)

    vorts3d = vcat([vorts_xslice, vorts_yslice, vorts_zslice]...);
    return vorts3d
end

function vortInBall1!(vcx, vcy, vcz, Δvcx, Δvcy, Δvcz, kdtree, ϵ, f, search)
    vp = [vcx + Δvcx, vcy + Δvcy, vcz + Δvcz]
    p_idxs = inrange(kdtree, vp, ϵ)
    union!(f, Set(p_idxs))
    union!(search, Set(p_idxs))
end

function vortInBall2!(vcx, vcy, vcz, Δvcx, Δvcy, Δvcz, kdtree, ϵ, f, search)
    vp = [vcx + Δvcx, vcy + Δvcy, vcz + Δvcz]
    p_idxs = inrange(kdtree, vp, ϵ)
    setdiff!(p_idxs, f) 
    union!(f, Set(p_idxs))
    union!(search, Set(p_idxs))
end

function connect_vortex_points_3d_harmonic(
    vorts_3d :: Vector{Vector{Float64}}, 
    X :: Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}, 
    α :: Float64, 
    N :: Int, 
    periodic_x,
    periodic_y,
    periodic_z
    ) :: Vector{Set{Int64}}

    @assert size(vorts_3d)[1] != 4

    v_matrix = zeros(3, size(vorts_3d)[1])
    for i in 1:3
        for j in 1:size(vorts_3d)[1]
            v_matrix[i, j] = vorts_3d[j][i]
        end
    end

    kdtree = KDTree(v_matrix)
    num_vorts = length(v_matrix[1,:])
    unvisited = Set(collect(1:num_vorts))
    fils = []
    x = X[1]; y = X[2]; z = X[3];
    Δx = x[2]-x[1]; Δy = y[2]-y[1]; Δz = z[2]-z[1];
    xdist = x[end]-x[1]; ydist = y[end]-y[1]; zdist = z[end]-z[1];

    if N == 1
        ϵ = (1+α)*sqrt(Δx^2+Δy^2+Δz^2)/N # 
    else 
        ϵ = (1+α)*sqrt(Δx^2+Δy^2+Δz^2)/(N-1) # 
    end

    if ϵ < Δx/3
        ϵ = Δx/3
    end

    while length(unvisited) > 0
        idx = first(unvisited)
        vc = v_matrix[:, idx]
        f_idxs = inrange(kdtree, vc, ϵ)
        f = Set(f_idxs)
        search = Set(f_idxs)
        setdiff!(search, idx)

        vcx = v_matrix[1,idx]; vcy=v_matrix[2,idx]; vcz = v_matrix[3,idx];
        if periodic_x
            if abs(vcx - x[1]) < ϵ
                vortInBall1!(vcx, vcy, vcz, xdist+Δx, 0, 0, kdtree, ϵ, f, search)
            elseif abs(vcx - x[end]) < ϵ
                vortInBall1!(vcx, vcy, vcz, -xdist-Δx, 0, 0, kdtree, ϵ, f, search)
            end
        end

        if periodic_y
            if abs(vcy - y[1]) < ϵ
                vortInBall1!(vcx, vcy, vcz, 0, ydist+Δy, 0, kdtree, ϵ, f, search)
            elseif abs(vcy - y[end]) < ϵ
                vortInBall1!(vcx, vcy, vcz, 0, -ydist-Δy, 0, kdtree, ϵ, f, search)
            end
        end
            
        if periodic_z
            if abs(vcz - z[1]) < ϵ
                vortInBall1!(vcx, vcy, vcz, 0, 0, zdist+Δz, kdtree, ϵ, f, search)
            elseif abs(vcz - z[end]) < ϵ
                vortInBall1!(vcx, vcy, vcz, 0, 0, -zdist-Δz, kdtree, ϵ, f, search)
            end
        end

        while length(search) > 0
            idx = first(search)
            setdiff!(search, idx)
            vc = v_matrix[:, idx]
            vc_idxs = inrange(kdtree, vc, ϵ)
            setdiff!(vc_idxs, f)
            union!(f, Set(vc_idxs))
            union!(search, Set(vc_idxs))
            vcx = v_matrix[1,idx]; vcy=v_matrix[2,idx]; vcz = v_matrix[3,idx];
            if periodic_x
                if abs(vcx - x[1]) < ϵ
                    vortInBall2!(vcx, vcy, vcz, xdist+Δx, 0, 0, kdtree, ϵ, f, search)
                elseif abs(vcx - x[end]) < ϵ
                    vortInBall2!(vcx, vcy, vcz, -xdist-Δx, 0, 0, kdtree, ϵ, f, search)
                end
            end
            if periodic_y
                if abs(vcy - y[1]) < ϵ
                    vortInBall2!(vcx, vcy, vcz, 0, ydist+Δy, 0, kdtree, ϵ, f, search)
                elseif abs(vcy - y[end]) < ϵ
                    vortInBall2!(vcx, vcy, vcz, 0, -ydist-Δy, 0, kdtree, ϵ, f, search)
                end
            end
            if periodic_z
                if abs(vcz - z[1]) < ϵ
                    vortInBall2!(vcx, vcy, vcz, 0, 0, zdist+Δz, kdtree, ϵ, f, search)
                elseif abs(vcz - z[end]) < ϵ
                    vortInBall2!(vcx, vcy, vcz, 0, 0, -zdist-Δz, kdtree, ϵ, f, search)
                end
            end
        end
        if length(f) > N
            push!(fils, f)
        end
        setdiff!(unvisited, f)
    end
    return fils
end



N = 2
@time vorts_3d = find_vortex_points_3d_harmonic(psi[:, :, 1:end-1], [X[1], X[2], X[3][1:end-1]], N)
# vorts_3d = find_vortex_points_3d_harmonic(psi, X, N)



plot_iso(psi, X)
scatterVortsOnIso(vorts_3d, 0.04)

new_v1::Vector{Vector{Float64}} = []

zb = 4
for i in eachindex(vorts_3d)
    if (vorts_3d[i][3] > z[1+zb] && vorts_3d[i][3] < z[end-zb])
        push!(new_v1, vorts_3d[i])
    end
end

plot_iso(psi, X)
scatterVortsOnIso(new_v1)

@time vorts_class = connect_vortex_points_3d_harmonic(vorts_3d, X, 0.6, N, true, true, false)
plot_iso(psi, X, true, true)
scatterClassifiedVortices(vorts_class, vorts_3d, X, 0.02)

@time v_sort = sort_classified_vorts_3d(vorts_class, vorts_3d, X); 
plot_iso(psi, X, true, true)
periodicPlotting(v_sort, X, 1)




# extrema(X[3])

# using JLD2

@save "quenchslab3d.jld2" psi X 






@time vorts_3d = find_vortex_points_3d_harmonic(psi, X, 4)

vdmat = vorts3DMatrix(vorts_3d)

maximum(vdmat[:, 3])

plot_iso(psi, X)
scatterVortsOnIso(vorts_3d, 0.05)