using FourierGPE
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


#--- parameters
L = (30.,30.,8.)
N = (128,128,32)
sim = Sim(L,N)
@unpack_Sim sim;


#--- Initialize simulation
γ = 0.05
μ = 25.0
tf = 2.5/γ
Nt = 200
t = LinRange(0.,tf,Nt)

# potential
import FourierGPE.V
V(x,y,z,t) = 4*z^2


#--- random initial state
x,y,z = X
ψi = randn(N)+im*randn(N)
ϕi = kspace(ψi,sim)

@pack_Sim! sim;

#--- Evolve in k space
@time sol = runsim(sim); # will take a few minutes to run.

psi = xspace(sol[60], sim)


plot_iso(psi, X, true, true)

psi_dense = dense(psi)

using JLD2

@save "quenchslab3d.jld2" psi X 

psi = xspace(sol[200], sim)

@save "quenchslab_t200.jld2" psi X 
