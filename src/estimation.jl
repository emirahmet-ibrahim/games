module Estimation

include("games.jl")
include("models.jl")

using .Games, .Models, BlackBoxOptim
using Statistics: mean, std
using Combinatorics: combinations
using Printf: @printf

export NLL, MSD, MAE, Brier, AIC, BIC, ENLL, EBrier, obj, fit, train, Vuong
export NEE, QRE, NI, LK1, LKr, QLK, HNI1, HNIr
export CH1, CHr, GCH1, GCHr, QCH1, QCHr, LM1, LMr, QPLK
export level0, poisson, hardmax, belief, Random
export SLK1, SLKr, SCH1, SCHr
export Game, G2, G3

## Loss function
function Loss(loss::Function, θ, model::Function, G::Game, games)
    L = 0.0;
    for i in games
        p = model(G.u, G.A[i], θ);
        p = max.(1e-12, p);
        L += loss(G.f[i], p);
    end
    return L
end

# Negative log-likelihood function
nloglike(f::Vector, p::Vector) = -sum(f .* log.(p));
NLL(θ, model::Function, G::Game, games) = Loss(nloglike, θ, model, G, games);

# Mean squared deviations
msd(f::Vector, p::Vector) = mean(abs2, f/sum(f) - p) * 1e+4;
MSD(θ, model::Function, G::Game, games) = Loss(msd, θ, model, G, games)/length(games);

# Mean absolute error
mae(f::Vector, p::Vector) = mean(abs, f/sum(f) - p) * 1e+2;
MAE(θ, model::Function, G::Game, games) = Loss(mae, θ, model, G, games)/length(games);

# Brier score
function brier(f::Vector, p::Vector)
    n = length(f);
    score = 0.0;
    for i = 1:n
        e = [Float64(i==j) for j=1:n];
        score += f[i] * sum(abs2, e - p) * 1e4;
    end
    return score / sum(f);
end
Brier(θ, model::Function, G::Game, games) = Loss(brier, θ, model, G, games)/length(games);

AIC(nlogL, k) = @. 2(nlogL + k);
BIC(nlogL, k) = 2nlogL + k * log.([10, 10, 20]);

## NNL for empirical choice probabilities (minimum NNL)
function ENLL(G::Game, games)
    L = 0.0;
    for i in games
        p = max.(1e-12, G.f[i]/sum(G.f[i]));
        L += nloglike(G.f[i], p);
    end
    return L
end
# Brier score for empirical choice probabilities
function EBrier(G::Game, games)
    L = 0.0;
    for i in games
        p = max.(1e-12, G.f[i]/sum(G.f[i]));
        L += brier(G.f[i], p);
    end
    return L / length(games);
end

## Search range for model parameters: model => (θ, search range, population size)
Params = Dict{Function, Tuple{String, Vector{Tuple{Float64, Float64}}, Int}}();
push!(Params, NEE => ("ϵ", [(0,1)], 200));
push!(Params, QRE => ("λ", [(0,10)], 200));
push!(Params, LKr => ("ϕ₀, ϕ₁, ϕ₂, ρ", [(0,1), (0,1), (0,1), (0,1)], 200));
push!(Params, LK1 => ("ϕ₀, ϕ₁, ϕ₂", [(0,1), (0,1), (0,1)], 200));
push!(Params, CHr => ("τ, ρ", [(0,20), (0,1)], 500));
push!(Params, CH1 => ("τ", [(0,20)], 500));
push!(Params, NI => ("λ, δ", [(0,10), (0,1)], 500));
push!(Params, QLK => ("ϕ₀, ϕ₁, ϕ₂, λ₁, μ, λ₂", [(0,1), (0,1), (0,1),
                                    (0,100), (0,100), (0,10_000)], 2000));
push!(Params, QCHr => ("τ, ρ, λ₁, λ₂", [(0,20), (0,1), (0,100), (0,100)], 3000));
push!(Params, QCH1 => ("τ, λ₁, λ₂", [(0,20), (0,100), (0,100)], 2000));
push!(Params, HNIr => ("τ, ρ, λ", [(0,30), (0,1), (0,10)], 1000));
push!(Params, HNI1 => ("τ, λ", [(0,30), (0,10)], 1000));
push!(Params, GCHr => ("τ, ρ, α", [(0,20), (0,1), (1,10)], 1000));
push!(Params, GCH1 => ("τ, α", [(0,20), (1,10)], 1000));
push!(Params, LMr => ("τ, ρ", [(0,20), (0,1)], 1000));
push!(Params, LM1 => ("τ", [(0,20)], 1000));
push!(Params, QPLK => ("τ, λ₁, λ₂, λ₃", [(0,20), (0,100), (0,100), (0,100)], 1000));

push!(Params, SLKr => ("ϕ₀, ϕ₁, ϕ₂, ρ₀, ρ₁, λ₁, λ₂", [(0,1), (0,1), (0,1),
                                    (0,1), (0,1), (0,100), (0,10_000)], 1000));
push!(Params, SLK1 => ("ϕ₀, ϕ₁, ϕ₂, ρ₁, λ₁, λ₂", [(0,1), (0,1), (0,1),
                                    (0,1), (0,100), (0,10_000)], 1000));
push!(Params, SCHr => ("ϕ₀, ϕ₁, ϕ₂, ρ, λ₁, λ₂", [(0,1), (0,1), (0,1),
                                    (0,1), (0,100), (0,10_000)], 1000));
push!(Params, SCH1 => ("ϕ₀, ϕ₁, ϕ₂, λ₁, λ₂", [(0,1), (0,1), (0,1),
                                    (0,100), (0,10_000)], 1000));

# Objective function for bboptimize: 2-or-3-player games
function obj(model::Function, loss::Function, G::Game; games=1:10)
    if model == QRE
        return θ -> loss(θ, (u,a,λ)->QRE(u,G.du,a,λ), G, games)
    else
        return θ -> loss(θ, model, G, games)
    end
end
# Objective function for bboptimize: all-player games
function obj(model::Function, loss::Function, G2::Game, G3::Game;
                games::Tuple=(1:10, 1:10))
    if loss == NLL
        return θ -> (obj(model, loss, G2; games=games[1])(θ) +
                     obj(model, loss, G3; games=games[2])(θ));
    else
        return θ -> (obj(model, loss, G2; games=games[1])(θ) +
                     obj(model, loss, G3; games=games[2])(θ))/2;
    end
end

## Statistical fit
function fit(model::Function, loss::Function, G::Vararg{Game}; Params=Params)
    θ, srange, psize = Params[model];
    m = (length(G) == 1) ? G[1].m : "all"
    modelname = nameof(model);  lossname = nameof(loss);
    msteps = 1000 * psize;
    res = bboptimize(obj(model, loss, G...); SearchRange = srange,
                        PopulationSize = psize, MaxSteps = msteps);
    θhat = best_candidate(res);
    if model in [LKr, LK1, QLK, SLKr, SLK1, SCHr, SCH1]
        θhat[1:3] ./= sum(θhat[1:3]);
    end
    open("./statistical_fit/$modelname.txt","a") do io
        println(io, "------$modelname, $m-player games, $lossname------")
        println(io, "optimizer = adaptive_de_rand_1_bin_radiuslimited")
        println(io, "SearchRange = $srange")
        println(io, "PopulationSize = $psize, MaxSteps = $msteps")
        println(io, "")
        println(io, "[$θ] = ", θhat)
        println(io, lossname, " = ", best_fitness(res))
        println(io, "")
    end
    return round(best_fitness(res), digits=2)
end

## Out-of-sample estimation:
# 5-5 split for 2-or-3-player games
# 9+9 training & 1+1 test for all games
complement(v::Union{Int, Vector{Int}}) = setdiff(1:10, v);
complement(v::Tuple{Int,Int}) = (complement(v[1]), complement(v[2]));

function train(model::Function, loss::Function, G::Vararg{Game}; Params=Params)
    θ, srange, psize = Params[model];
    m = (length(G) == 1) ? G[1].m : "all"
    modelname = nameof(model);  lossname = nameof(loss);
    msteps = 1000 * psize;

    if length(G) == 1
        games = collect(combinations(1:10, 5));
    else
        games = Iterators.product(1:10, 1:10);
    end
    train_loss = zeros(length(games));
    test_loss = zeros(length(games));
    open("./out_of_sample/$modelname-$m-player-$lossname.txt","a") do io
        println(io, "----------$modelname, $m-player, $lossname----------")
        println(io, "optimizer = adaptive_de_rand_1_bin_radiuslimited")
        println(io, "SearchRange = $srange")
        println(io, "PopulationSize = $psize, MaxSteps = $msteps")
        println(io, "")
        println(io, "iter test_games  train     test       $θ ")
        for (i, test_games) in enumerate(games)
            println("iter $i")
            train_games = complement(test_games);
            res = bboptimize(obj(model, loss, G...; games=train_games);
                    SearchRange = srange, PopulationSize = psize, MaxSteps = msteps);
            θhat = best_candidate(res);
            train_loss[i] = best_fitness(res);
            test_loss[i] = obj(model, loss, G...; games=test_games)(θhat);
            if model in [LKr, LK1, QLK, SLKr, SLK1, SCHr, SCH1]
                θhat[1:3] ./= sum(θhat[1:3]);
            end

            @printf(io, "%3i ", i)
            for g in test_games; @printf(io, "%2i", g) end
            @printf(io, "%  8.4f ", round(train_loss[i], digits=4))
            @printf(io, "% 8.4f ", round(test_loss[i], digits=4))
            println(io, "θ = ", round.(θhat, digits=4))
        end
        println(io, "")
        println(io, "mean(train_loss) = ", mean(train_loss))
        println(io, "mean(test_loss) = ", mean(test_loss))
        println(io, "std(train_loss) = ", std(train_loss))
        println(io, "std(test_loss) = ", std(test_loss))
    end
    return round(mean(train_loss), digits=2), round(mean(test_loss), digits=2)
end

## Vuong closeness test
function Vuong(nll::Matrix{T}, d::Vector{Int}) where T
# nnl[i,j] : NNL in game i for model j
# d[j] : number of parameters in model j
    n, m = size(nll);  # number of games & models
    Z = zeros(T, m, m);
    S = [sum(nll[:,j]) for j=1:m];
    for j = 1:m
        for k = 1:m
            if j == k; continue; end
            #ω2 = var(nll[:,j] - nll[:,k]);
            ω2 = sum(abs2, nll[:,j] - nll[:,k] .- (S[j]-S[k])/n) / (n-1);
            Z[j,k] = (S[j] - S[k] + (d[j]-d[k])*log(n)/2) / sqrt(n * ω2);
        end
    end
    return round.(Z, digits=2)
end

end # of module
