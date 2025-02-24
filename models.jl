module Models

using LinearAlgebra
using NNlib: softmax

export NEE, QRE, NI, LK1, LKr, QLK, HNI1, HNIr
export CH1, CHr, GCH1, GCHr, QCH1, QCHr, LM1, LMr
export level0, poisson, hardmax, belief, Random
export SLK1, SLKr, SCH1, SCHr, QPLK

## Helper functions
# Level-0 behavior
function level0(ρ::Real, n::Int)
    p = ρ * ones(n) / n;  # level-0 chooses randomly with probability ρ
    p[1] += 1-ρ;          # level-0 chooses 18 with probability 1-ρ
    return p
end

poisson(k::Int64, τ::Float64) = τ^k * exp(-τ) / factorial(k);

function hardmax(u::Vector{<:Real})
    v = u .> maximum(u)-1e-8;
    return v / sum(v)
end

# level k's belief about the relative proportion of lower levels in GCH
function belief(f::Vector{<:Real}, α::Real=1)
    g = f.^α;
    return g / sum(g)
end

# Computes a root of F using Newton's method
function Newton(F::Function, DF::Function, p::Vector; ϵ::Real=1e-8,
                maxiter::Int=1000, guarded::Bool=true, s_min::Real=1e-12)
    for k = 1:maxiter
        d = -DF(p) \ F(p);
        s = 1.0;
        if guarded
            while norm(F(p+s*d)) > (1-s/2)*norm(F(p))
                s /= 2;
                if s < s_min
                    println("line search failure")
                    return p;
                end
            end
        end
        p .+= s*d;
        if norm(F(p)) < ϵ
            break;
        end
    end
    #println("maxiter has been reached!")
    return p
end

## Quantal Response Equilibrium
function QRE(u::Function, du::Function, a::Vector{Int}, λ::Real)
    F(p) = p - softmax(λ * u(a, p));
    DF(p) = begin
        s = softmax(λ * u(a, p));
        return I - λ * (diagm(s) - s*s') * du(a, p);
    end
    p = belief(a, λ);  # initial guess
    p = Newton(F::Function, DF::Function, p::Vector);
    norm(F(p)) < 1e-8 || println("|F(p)| = ", norm(F(p)))
    return p
end
QRE(u::Function, du::Function, a::Vector{Int}, θ::Vector{<:Real}) =
    QRE(u, du, a, θ[1]);

## Noisy Introspection
function NI(u::Function, a::Vector{Int}, θ::Vector{<:Real}; kmax::Int=10)
    n = length(a);
    p = ones(n) / n;  # choosing uniformly random
    for k = kmax:-1:0
        λ = θ[1] * θ[2]^k;
        p = softmax(λ * u(a, p));
    end
    return p
end

## Level-k
function LK(u::Function, a::Vector{Int}, f::Vector{<:Real}, ρ)
    n = length(a);
    kmax = length(f) - 1;
    P = zeros(n, kmax+1);  # level-k player chooses a[i] with prob. P[i,k+1]
    P[:,1] = level0(ρ, n);
    for k = 1:kmax
        P[:,k+1] = hardmax(u(a, P[:,k]));
    end
    return P * f / sum(f)
end

LK1(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = LK(u, a, θ, 1);
LKr(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = LK(u, a, θ[1:end-1], θ[end]);

## Quantal Level-k
function QLK(u::Function, a::Vector{Int}, f::Vector{<:Real}, λ::Vector{<:Real})
    n = length(a);
    P = zeros(n, 3);  # level-k chooses a[i] with probability P[i,k+1]
    P[:,1] = level0(1, n);                 # level-0 chooses uniformly random
    P[:,2] = softmax(λ[1] * u(a, P[:,1])); # level-1 quantal respond to level-0
    q = softmax(λ[2] * u(a, P[:,1]));      # level-2's belief
    P[:,3] = softmax(λ[3] * u(a, q)); # level-2 quantal respond to q
    return P * f / sum(f)
end

QLK(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = QLK(u, a, θ[1:3], θ[4:6]);

## (Generalized) Cognitive Hierarchy
function CH(u::Function, a::Vector{Int}, τ::Real, ρ::Real; α::Real=1, kmax::Int=10)
    n = length(a);
    f = poisson.(0:kmax, τ);
    P = zeros(n, kmax+1);  # level-k player chooses a[i] with prob. P[i,k+1]
    P[:,1] = level0(ρ, n);
    for k = 1:kmax
        g = belief(f[1:k], α); # level-k's belief about the dist. of players
        p = P[:,1:k] * g;
        P[:,k+1] = hardmax(u(a, p));
    end
    return P * f / sum(f)
end

CH1(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = CH(u, a, θ[1], 1);
CHr(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = CH(u, a, θ[1], θ[2]);

GCH1(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = CH(u, a, θ[1], 1; α=θ[2]);
GCHr(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = CH(u, a, θ[1], θ[2]; α=θ[3]);

## Quantal Cognitive Hierarchy
function QCH(u::Function, a::Vector{Int}, τ::Real, ρ::Real, λ::Vector{<:Real};
                kmax::Int=2)
    n = length(a);
    f = poisson.(0:kmax, τ);
    P = zeros(n, kmax+1);  # level-k player chooses a[i] with prob. P[i,k+1]
    P[:,1] = level0(ρ, n);
    for k = 1:kmax
        g = belief(f[1:k]);  # level-k's belief about the dist. of players
        p = P[:,1:k] * g;
        P[:,k+1] = softmax(λ[k] * u(a, p));
    end
    return P * f / sum(f)
end

QCH1(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = QCH(u, a, θ[1], 1.0, θ[2:end]);
QCHr(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = QCH(u, a, θ[1], θ[2], θ[3:end]);

## Level-m
function LM(u::Function, a::Vector{Int}, τ::Real, ρ::Real; kmax::Int=10)
    n = length(a);
    f = poisson.(0:kmax, τ);
    P = zeros(n, kmax+1);  # level-k player chooses a[i] with prob. P[i,k+1]
    P[:,1] = level0(ρ, n);
    for k = 1:kmax
        g = hardmax(f[1:k]);  # level-k's belief about the dist. of players
        p = P[:,1:k] * g;
        P[:,k+1] = hardmax(u(a, p));
    end
    return P * f / sum(f)
end

LM1(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = LM(u, a, θ[1], 1.0);
LMr(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = LM(u, a, θ[1], θ[2]);

## Heterogeneous Noisy Introspection
function HNI(u::Function, a::Vector{Int}, τ::Real, ρ::Real, λ::Real; kmax::Int=10)
    n = length(a);
    f = poisson.(0:kmax, τ);
    P = zeros(n, kmax+1);  # level-k player chooses a[i] with prob. P[i,k+1]
    P[:,1] = level0(ρ, n); # choosing uniformly random
    for k = 1:kmax
        P[:,k+1] = softmax(λ * u(a, P[:,k]));
    end
    return P * f / sum(f)
end

HNI1(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = HNI(u, a, θ[1], 1.0, θ[2]);
HNIr(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = HNI(u, a, θ[1], θ[2], θ[3]);

## Quantal Poisson Level-k
function QPLK(u::Function, a::Vector{Int}, τ::Real, λ::Vector{<:Real}; kmax::Int=3)
    n = length(a);
    f = poisson.(0:kmax, τ);
    P = zeros(n, kmax+1);  # level-k player chooses a[i] with prob. P[i,k+1]
    P[:,1] = level0(1, n);
    for k = 1:kmax
        P[:,k+1] = softmax(λ[k] * u(a, P[:,k]));
    end
    return P * f / sum(f)
end

QPLK(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = QPLK(u, a, θ[1], θ[2:end]);

## Random
function Random(u::Function, a::Vector{Int}, θ=0)
    n = length(a);
    return ones(n) / n;
end

## Nash Equilibrium
p_nash = Dict{Tuple{Int,Vector{Int}}, Vector{<:Real}}();
push!(p_nash, (2, [18,12]) => [4/5, 1/5]);
push!(p_nash, (2, [18,12,12]) => [3/4, 1/8, 1/8]);
push!(p_nash, (2, [18,12,12,12]) => [8/11, 1/11, 1/11, 1/11]);
push!(p_nash, (2, [18,12,12,12,12]) => [5/7, 1/14, 1/14, 1/14, 1/14]);
push!(p_nash, (2, [18,10]) => [13/14, 1/14]);
push!(p_nash, (2, [18,10,10]) => [21/23, 1/23, 1/23]);
push!(p_nash, (2, [18,10,10,10,10]) => [37/41, 1/41, 1/41, 1/41, 1/41]);
push!(p_nash, (2, [18,14]) => [11/16, 5/16]);
push!(p_nash, (2, [18,14,14]) => [3/5, 1/5, 1/5]);
push!(p_nash, (2, [18,14,14,14,14]) => [23/43, 5/43, 5/43, 5/43, 5/43]);

p12(n) = (sqrt(45n^2 + 36n + 12) - 3(n+2))/2(3n^2 - 2); # n is # of 12's
push!(p_nash, (3, [18,12,6]) => [1-p12(1), p12(1), 0]);
push!(p_nash, (3, [18,12,9]) => [0.6623006439107122,
                                 0.3051735202178473,
                                 0.03252583587144042]);
push!(p_nash, (3, [18,12,12]) => [1-2p12(2), p12(2), p12(2)]);
push!(p_nash, (3, [18,12,12,12]) => [1-3p12(3), p12(3), p12(3), p12(3)]);
push!(p_nash, (3, [18,12,12,12,12]) => [1-4p12(4), p12(4), p12(4), p12(4), p12(4)]);
push!(p_nash, (3, [18,12,12,12,12,12]) => [1-5p12(5), p12(5), p12(5), p12(5), p12(5), p12(5)]);
push!(p_nash, (3, [18,12,9,9]) => [0.6548058664811155, 0.2972823752010170,
                                   0.02395587915893378, 0.02395587915893378]);
push!(p_nash, (3, [18,12,9,9,9]) => [0.6504450939150311, 0.2926825986180931,
                                     0.01895743582229193, 0.01895743582229193,
                                     0.01895743582229193]);
push!(p_nash, (3, [18,12,12,9,9]) => [1-2p12(2), p12(2), p12(2), 0, 0]);
push!(p_nash, (3, [18,12,12,9]) => [1-2p12(2), p12(2), p12(2), 0]);

function NE(u::Function, a::Vector{Int}, θ=0; p=p_nash)
    m = u([18],[1])[1] ≈ 9 ? 2 : 3;
    return p[(m, a)]
end

## Nash Equilibrium with error
function NEE(u::Function, a::Vector{Int}, θ::Vector{<:Real})
    p_nash = NE(u, a);
    p_rand = Random(u, a);
    return θ[1] * p_rand + (1-θ[1]) * p_nash
end

## SLK (Not included in the paper)
function SLK(u::Function, a::Vector{Int}, f::Vector{<:Real}, ρ₀::Real, ρ₁::Real,
                λ::Vector{<:Real})
    n = length(a);
    P = zeros(n, 3);  # level-k chooses a[i] with probability P[i,k+1]
    P[:,1] = level0(ρ₀, n);
    P[:,2] = softmax(λ[1] * u(a, P[:,1])); # level-1 quantal respond to level-0
    q = ρ₁ * P[:,1] + (1-ρ₁) * P[:,2]; # level-2's belief
    P[:,3] = softmax(λ[2] * u(a, q)); # level-2 quantal respond to q
    return P * f / sum(f)
end

SLK1(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = SLK(u, a, θ[1:3], 1.0, θ[4], θ[5:6]);
SLKr(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = SLK(u, a, θ[1:3], θ[4], θ[5], θ[6:7]);


## SCH (Not included in the paper)
function SCH(u::Function, a::Vector{Int}, f::Vector{<:Real}, ρ::Real,
                λ::Vector{<:Real})
    n = length(a);
    P = zeros(n, 3);  # level-k chooses a[i] with probability P[i,k+1]
    P[:,1] = level0(ρ, n);
    P[:,2] = softmax(λ[1] * u(a, P[:,1])); # level-1 quantal respond to level-0
    q = P[:,1:2] * belief(f[1:2]); # level-2's belief
    P[:,3] = softmax(λ[2] * u(a, q)); # level-2 quantal respond to q
    return P * f / sum(f)
end

SCH1(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = SCH(u, a, θ[1:3], 1, θ[4:5]);
SCHr(u::Function, a::Vector{Int}, θ::Vector{<:Real}) = SCH(u, a, θ[1:3], θ[4], θ[5:6]);

end # of module
