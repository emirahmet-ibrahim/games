module Games

using LinearAlgebra: diagm
export Game, G2, G3

struct Game
    m::Int64                      # m ∈ {2,3}: the number of players
    A::Vector{Vector{Int64}}      # A[j]     : boxes in game j
    u::Function                   # u(a,p)   : payoff function
    du::Function                  # du(a,p)  : Jacobian of u wrt p
    f::Vector{Vector{Float64}}    # f[j]     : observed frequencies in game j
end

# Payoff functions
u2(a::Vector{Int}, p::Vector{<:Real}) = @. a * (1 - p/2);
u3(a::Vector{Int}, p::Vector{<:Real}) = @. a * (1 - p * (1 - p/3));

# Jacobian of u with respect to p
du2(a::Vector{Int}, p::Vector{<:Real}) = diagm(-a/2);
du3(a::Vector{Int}, p::Vector{<:Real}) = diagm(@. a * (2p/3 - 1));

## Two-player games
A2 = [[18,12],           # game 1
      [18,12,12],        # game 2
      [18,12,12,12],     # game 3
      [18,12,12,12,12],  # game 4
      [18,10],           # game 5
      [18,10,10],        # game 6
      [18,10,10,10,10],  # game 7
      [18,14],           # game 8
      [18,14,14],        # game 9
      [18,14,14,14,14]]; # game 10

f2 = [[64,16],                   # game 1: [18,12]
      [56,12,12],                # game 2: [18,12,12]
      [43,37/3,37/3,37/3],       # game 3: [18,12,12,12]
      [48,8,8,8,8],              # game 4: [18,12,12,12,12]
      [69,11],                   # game 5: [18,10]
      [62,9,9],                  # game 6: [18,10,10]
      [57,23/4,23/4,23/4,23/4],  # game 7: [18,10,10,10,10]
      [55,25],                   # game 8: [18,14]
      [49,31/2,31/2],            # game 9: [18,14,14]
      [30,25/2,25/2,25/2,25/2]]; # game 10: [18,14,14,14,14]

G2 = Game(2, A2, u2, du2, f2);

## Three-player games
A3 = [[18,12,6],           # game 1
      [18,12,9],           # game 2
      [18,12,12],          # game 3
      [18,12,12,12],       # game 4
      [18,12,12,12,12],    # game 5
      [18,12,12,12,12,12], # game 6
      [18,12,9,9],         # game 7
      [18,12,9,9,9],       # game 8
      [18,12,12,9,9],      # game 9
      [18,12,12,9]];       # game 10

f3 = [[62,17,2],                # game 1: [18,12,6]
      [48,25,8],                # game 2: [18,12,9]
      [58,23/2,23/2],           # game 3: [18,12,12]
      [42,13,13,13],            # game 4: [18,12,12,12]
      [35,23/2,23/2,23/2,23/2], # game 5: [18,12,12,12,12]
      [36,9,9,9,9,9],           # game 6: [18,12,12,12,12,12]
      [45,26,5,5],              # game 7: [18,12,9,9]
      [45,20,16/3,16/3,16/3],   # game 8: [18,12,9,9,9]
      [49,7,7,9,9],             # game 9: [18,12,12,9,9]
      [54,27/2,27/2,0]];        # game 10: [18,12,12,9]

G3 = Game(3, A3, u3, du3, f3);

end # of module
