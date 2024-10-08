

function find_vortex_points_3d(
    psi :: Array{ComplexF64, 3}, 
    X :: Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}, 
    N :: Int = 1
    ) :: Vector{Vector{Float64}}
    # TODO: Add periodic checks 
    @assert N <= 16
    @assert N >= 1
    # print("inner: " + N)

    x = X[1]; y = X[2]; z = X[3];
    dx = x[2]-x[1]; dy = y[2]-y[1]; dz = z[2]-z[1];


    @assert size(psi)[1] == length(x)
    @assert size(psi)[2] == length(y)
    @assert size(psi)[3] == length(z)

    x_itp = interpolate(X[1], BSpline(Linear()));
    y_itp = interpolate(X[2], BSpline(Linear()));
    z_itp = interpolate(X[3], BSpline(Linear()));

    x_etp = extrapolate(x_itp, Line())
    y_etp = extrapolate(y_itp, Line())
    z_etp = extrapolate(z_itp, Line())

    psi_itp = interpolate(psi, BSpline(Quadratic(Periodic(OnCell()))))
    psi_etp = extrapolate(psi_itp, Periodic())

    x_range = LinRange(-1,length(x)+2,N*(length(x)+4))
    y_range = LinRange(-1,length(y)+2,N*(length(y)+4))
    z_range = LinRange(-1,length(z)+2,N*(length(z)+4))

    x = LinRange(x[1]-2*dx, x[end]+2*dx, length(x)+4);
    y = LinRange(y[1]-2*dy, y[end]+2*dy, length(y)+4);
    z = LinRange(z[1]-2*dz, z[end]+2*dz, length(z)+4);

    ## loop vectorisation, run in parallel 
    vorts3d = []
    vorts_xslice = []
    vorts_yslice = []
    vorts_zslice = []

    results_x = [[] for _ in 1:Threads.nthreads()]
    results_y = [[] for _ in 1:Threads.nthreads()]
    results_z = [[] for _ in 1:Threads.nthreads()]

    let z = z, y=y
        # Threads.@threads :static for xidx in x_range
        # Threads.@threads for xidx in x_range
        # for xidx in x_range
        @floop for xidx in x_range
            vorts_x = vortex_array(findvortices(Torus(psi_etp(xidx, y_range[1]:y_range[end], z_range[1]:z_range[end]), y, z)))
            for vidx_x in 1:size(vorts_x)[1]
                v_x = vorts_x[vidx_x, :]
                vx_x = [x_etp(xidx), v_x[1], v_x[2], v_x[3]]
                # push!(vorts_xslice, vx_x)
                push!(results_x[Threads.threadid()], vx_x)
            end
        end
    end

    let x=x, z=z
        # Threads.@threads :static for yidx in y_range
        # Threads.@threads for yidx in y_range
        # @sync @floop for yidx in y_range
        # for yidx in y_range
        @floop for yidx in y_range
            vorts_y = vortex_array(findvortices(Torus(psi_etp(x_range[1]:x_range[end], yidx, z_range[1]:z_range[end]), x, z)))
            for vidx_y in 1:size(vorts_y)[1]
                # if isdefined(vorts, vidx)
                v_y = vorts_y[vidx_y, :]
                vy_y = [v_y[1], y_etp(yidx), v_y[2], v_y[3]]
                # push!(vorts_yslice, vy_y)
                push!(results_y[Threads.threadid()], vy_y)
                # end
            end
        end
    end

    let x=x, y=y
        # Threads.@threads :static for zidx in z_range
        # Threads.@threads for zidx in z_range
        # for zidx in z_range
        @floop for zidx in z_range
            local vorts_z = vortex_array(findvortices(Torus(psi_etp(x_range[1]:x_range[end], y_range[1]:y_range[end], zidx), x, y)))
            for vidx_z in 1:size(vorts_z)[1]

                v_z = vorts_z[vidx_z, :]
                vz_z = [v_z[1], v_z[2], z_etp(zidx), v_z[3]]

                # push!(vorts_zslice, vz_z)
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

function connect_vortex_points_3d(
    vorts_3d :: Vector{Vector{Float64}}, 
    X :: Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}, 
    α :: Float64, 
    N :: Int, 
    periodic :: Bool =false
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
        if periodic
            vcx = v_matrix[1,idx]; vcy=v_matrix[2,idx]; vcz = v_matrix[3,idx];

            if abs(vcx - x[1]) < ϵ
                vortInBall1!(vcx, vcy, vcz, xdist+Δx, 0, 0, kdtree, ϵ, f, search)
            elseif abs(vcx - x[end]) < ϵ
                vortInBall1!(vcx, vcy, vcz, -xdist-Δx, 0, 0, kdtree, ϵ, f, search)
            end

            if abs(vcy - y[1]) < ϵ
                vortInBall1!(vcx, vcy, vcz, 0, ydist+Δy, 0, kdtree, ϵ, f, search)
            elseif abs(vcy - y[end]) < ϵ
                vortInBall1!(vcx, vcy, vcz, 0, -ydist-Δy, 0, kdtree, ϵ, f, search)
            end

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
            if periodic
                vcx = v_matrix[1,idx]; vcy=v_matrix[2,idx]; vcz = v_matrix[3,idx];

                if abs(vcx - x[1]) < ϵ
                    vortInBall2!(vcx, vcy, vcz, xdist+Δx, 0, 0, kdtree, ϵ, f, search)
                elseif abs(vcx - x[end]) < ϵ
                    vortInBall2!(vcx, vcy, vcz, -xdist-Δx, 0, 0, kdtree, ϵ, f, search)
                end

                if abs(vcy - y[1]) < ϵ
                    vortInBall2!(vcx, vcy, vcz, 0, ydist+Δy, 0, kdtree, ϵ, f, search)
                elseif abs(vcy - y[end]) < ϵ
                    vortInBall2!(vcx, vcy, vcz, 0, -ydist-Δy, 0, kdtree, ϵ, f, search)
                end

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

function sort_classified_vorts_3d(
    v_class :: Vector{Set{Int64}}, 
    vorts_3d :: Vector{Vector{Float64}}, 
    X :: Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    ) :: Vector{Any}

    ## Paramaters of box
    x = X[1]; y = X[2]; z = X[3];
    dx = x[2]-x[1]; dy = y[2]-y[1]; dz = z[2]-z[1];
    Lx = x[end]-x[1] + dx; Ly = y[end]-y[1] + dy; Lz = z[end]-z[1] + dz;

    # All vortex points in matrix form
    v_matrix = vcat(vorts_3d'...)[:,1:3]'

    # An array of arrays for each vortex sorted
    vorts_sorted = []

    # Loop through the classified vortices
    for i in 1:length(v_class)

        vi_set = v_class[i] # Set of column indicies of v_matrix for this vortex
        vi_mat = v_matrix[:, collect(vi_set)] # Create v_matrix for this vortex
        vi_mat = vi_mat[:, [vortInBounds3(vi_mat[:, j], X) for j = 1:size(vi_mat, 2)]] # Filters half of vortices outside bounds of box
        tree = BallTree(vi_mat, PeriodicEuclidean([Lx, Ly, Lz])) # BallTree
        vi_set = Set(collect(1:size(vi_mat, 2))) # Set of indicies for current v_matrix
        vi_sorted_index = [] # Indicies of vortices in order
        
        idx = first(vi_set) # Pop the first vortex
        vc = vi_mat[:, idx]
        setdiff!(vi_set, idx) # Take it away from the index set
        push!(vi_sorted_index, idx) # Push it to start of list
        
        # Find the nearest neighbour to vc excluding vortices found (not in vi_set),
        # push to vi_sorted_index and take away from vi_set, then set vc to the vortex found.
        # Finish when no vortex found near to vc within dx distance
        while length(vi_set) != 0
            v_new_index, dist = nn(tree, vc, i -> i ∉ vi_set)
            if dist < dx
                push!(vi_sorted_index, v_new_index)
                setdiff!(vi_set, v_new_index)
                vc = vi_mat[:, v_new_index]
            else
                break
            end
        end

        vi_sorted = vi_mat[:, collect(vi_sorted_index)] # Get the matrix of points in sorted order
        vi_sorted_array = collect(eachcol(vi_sorted)) # Convert into array of points
        vi_sorted_array = vi_sorted_array[[vortInBounds(vi_sorted_array[j], X) for j = 1:length(vi_sorted_array)]] # Filter all vortices not on grid
        push!(vorts_sorted, vi_sorted_array) # Append to the vorts_sorted array
    end
    @assert length(vorts_sorted) == length(v_class)
    return vorts_sorted
end

# non-exported helper funcs
function vortInBounds(v, X)
    x = X[1]; y = X[2]; z = X[3];
    dx = x[2]-x[1]; dy = y[2]-y[1]; dz = z[2]-z[1];
    if ((v[1] >= x[1]) && (v[1] <= x[end]) && 
        (v[2] >= y[1]) && (v[2] <= y[end]) && 
        (v[3] >= z[1]) && (v[3] <= z[end]))
        return true
    else 
        return false
    end
end


function vortInBounds3(v, X)
    x = X[1]; y = X[2]; z = X[3];
    dx = x[2]-x[1]; dy = y[2]-y[1]; dz = z[2]-z[1];
    if ((v[1] >= x[1]-dx) && (v[1] <= x[end]) && 
        (v[2] >= y[1]-dy) && (v[2] <= y[end]) && 
        (v[3] >= z[1]-dz) && (v[3] <= z[end]))
        return true
    else 
        return false
    end
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