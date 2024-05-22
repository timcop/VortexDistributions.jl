using VortexDistributions, GLMakie, Colors

using ColorSchemes

function plot_iso(psi, X, heal_2=false)
    density = abs2.(psi)
    pmax = maximum(density)
    density = density/pmax

    cbarPal= :plasma
    cmap = get(colorschemes[cbarPal], LinRange(0,1,100))
    cmap2 = [(cmap[i], 0.5) for i in 1:100]


    if heal_2
        # scene = volume(X[1], X[2], X[3], density, algorithm = :iso, visible=visible, isovalue=0.65,isorange=0.075, colormap=cmap2, transparency=true)
        scene = volume(X[1], X[2], X[3], density, algorithm = :iso, isovalue=0.65,isorange=0.075, colormap=cmap2, transparency=true)
    else
        # scene = volume(X[1], X[2], X[3], density, algorithm = :iso, visible=visible, colormap=cmap2, transparency=true, isovalue=0.1,isorange=0.075)
        scene = volume(X[1], X[2], X[3], density, algorithm = :iso, colormap=cmap2, transparency=true)
    end
    screen = display(scene)
    # resize!(screen, 2998, 1920)
end

function vorts3DMatrix(vorts)
    vorts = vcat(vorts'...)
end

function scatterVortsOnIso(vorts, markersize=0.1)
    vorts = vorts3DMatrix(vorts);
    
    meshscatter!(vorts[:, 1], vorts[:, 2], vorts[:, 3], color="blue", markersize=markersize)
end

function scatterClassifiedVortices(vortSets, vorts_3d, X, edges=false)
    colors = distinguishable_colors(length(vortSets),[RGB(0.8103465454545455,0.2951044545454546,0.4575856363636363)],dropseed=true)
    v_matrix = vcat(vorts_3d'...)[:,1:3]'

    for i in 1:length(vortSets)
        vi = v_matrix[:, collect(vortSets[i])]
        if !edges
            vi = vi[:, [vortInBounds(vi[:, i], X) for i = 1:length(vi[1, :])]] # Filters vortices that aren't on the grid
        end
        meshscatter!(vi[1,:],vi[2,:],vi[3,:],color=colors[i])
    end
end

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

function plot_line(vort_sort, color, linewidth)
    vx = [vort_sort[i][1] for i in 1:length(vort_sort)]
    vy = [vort_sort[i][2] for i in 1:length(vort_sort)]
    vz = [vort_sort[i][3] for i in 1:length(vort_sort)]

    lines!(vx, vy, vz, linewidth = linewidth, color = color)
end

function periodicPlotting(vorts_sorted, X, linewidth=5)
    x = X[1]; dx = x[2]-x[1];

    vorts_plotting = []

    for i in 1:length(vorts_sorted)
        vort_current = []
        vi = vorts_sorted[i]
        prev_break = 1
        for j in 1:length(vi)-1
            if euclid(vi[j], vi[j+1]) > dx
                push!(vort_current, vi[prev_break:j])
                prev_break = j+1
            end
        end
        push!(vort_current, vi[prev_break:end])
        if euclid(vort_current[1][1], vort_current[end][end]) < dx
            push!(vort_current[end], vort_current[1][1])
        end

        push!(vorts_plotting, vort_current)
    end
    
    colors = distinguishable_colors(length(vorts_plotting),[RGB(0.8103465454545455,0.2951044545454546,0.4575856363636363)],dropseed=true);
    for i in 1:length(vorts_plotting)
        vi = vorts_plotting[i]
        try
            for j in 1:length(vi)
                plot_line(vi[j], colors[i], linewidth)
            end
        catch
        end
    end
end

function euclid(v1, v2)
    @assert length(v1) == length(v2)
    sum = 0
    for i in 1:length(v1)
        sum += (v1[i]-v2[i])^2
    end
    sum = sqrt(sum)
    return sum
end

