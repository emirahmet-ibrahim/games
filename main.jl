# Main
cd("/Users/ibrahimemirahmetoglu/Desktop/games/code")

include("estimation.jl");
using .Estimation

# Statistical fit
for model in [NEE, QRE, LKr, LK1, CHr, CH1, NI, QLK, QCHr, QCH1,
                HNIr, HNI1, GCHr, GCH1, LMr, LM1, QPLK]
    res = Float64[];
    for G in [(G2,), (G3,), (G2, G3)]
        for loss in [NLL, MSD, MAE]
            best_fit = fit(model, loss, G...);
            append!(res, best_fit);
        end
    end
    println(nameof(model), " = ", res);
end

# Out-of-sample prediction
model = LK1;
res = Float64[];
for G in [(G2,), (G3,), (G2, G3)]
    for loss in [NLL, MSD, MAE]
        best_fit = train(model, loss, G...);
        append!(res, best_fit);
    end
end
println(nameof(model), " = ", res);

## Likelihood Ratio Test
using Distributions

pvalue(LR::Real, df::Int) = ccdf(Chisq(df), LR);
pvalue(z::Real) = z > 0 ? ccdf(Normal(), z) : cdf(Normal(), z);
#cquantile(Chisq(1), .005)

## Vuong closeness test
function nll(θ::Vector, model::Function, G::Game)
    if model == QRE
        return [NLL(θ, (u,a,λ)->QRE(u,G.du,a,λ), G, g) for g=1:10];
    else
        return [NLL(θ, model, G, g) for g=1:10];
    end
end

# Optimal parameters
nee2 = [0.22816630888893655];
nee3 = [0.18641717911628036];
neea = [0.20392067464305313];

qre2 = [1.2797941612338597];
qre3 = [1.295661930034289];
qrea = [1.2899302003501394];

lkr2 = [0.8484674301456708, 0.15153256985432925, 1.0762227589573408e-16,
        0.5187004553128608];
lkr3 = [0.473830500627248, 0.18440257332959992, 0.3417669260431522,
        0.6283764921930257];
lkra = [0.7998092065330853, 0.1929284731482002, 0.007262320318714708,
        0.3571428583333329];

ch12 = [10.201492826717626];
ch13 = [1.2521713700638517];
ch1a = [1.1568678956947573];

ni2 = [0.3032573782493102, 0.023349221077748672];
ni3 = [0.9464267261580317, 0.767176799248232];
nia = [0.9773590434854821, 0.7627009494120097];

qlk2 = [6.4149091870835324e-15, 0.9736862050538972, 0.02631379494609634,
        0.3024895977214658, 0.4247370698549853, 9974.48984286512];
qlk3 = [3.8319915955690136e-16, 0.920151146527204, 0.0798488534727958,
        0.3150922506154841, 78.07914775127283, 6214.658074993144];
qlka = [3.721981716662024e-16, 0.9234718843343905, 0.07652811566560923,
        0.2773843369106775, 0.2350537258692322, 6298.910211541035];

qch12 = [19.9999999999989, 5.492326843096685e-15, 0.3337542671990832];
qch13 = [19.999999999999712, 0.11716155903861109, 0.4952445874149458];
qch1a = [19.999999999999613, 0.0771968248460994, 0.4024116261870132];

qplk2 = [3.073252013105745, 0.19623113272005327,
         0.6469862447146864, 2.5361655775271834];
qplk3 = [9.599170882111459, 0.31459117306070356,
         2.527188904582883, 0.24872365557378495];
qplka = [19.999999999994674, 0.14469013795756658,
         0.366690728132235, 0.781992395820689];

## Models: Random, Emp, NEE, QRE, LKr, CH1, NI, QLK, QCH1, QPLK
d = [0, 0, 1, 1, 3, 1, 2, 5, 3, 4];  # number of parameters
NLL2 = zeros(10, 10);
NLL3 = zeros(10, 10);
NLLa = zeros(20, 10);

# Random
NLL2[:,1] = nll([0.0], Random, G2);
NLL3[:,1] = nll([0.0], Random, G3);
NLLa[:,1] = vcat(nll([0.0], Random, G2), nll([0.0], Random, G3));
# Empirical
NLL2[:,2] = [ENLL(G2, g) for g=1:10];
NLL3[:,2] = [ENLL(G3, g) for g=1:10];
NLLa[:,2] = vcat([ENLL(G2, g) for g=1:10], [ENLL(G3, g) for g=1:10]);
# NEE
NLL2[:,3] = nll(nee2, NEE, G2);
NLL3[:,3] = nll(nee3, NEE, G3);
NLLa[:,3] = vcat(nll(neea, NEE, G2), nll(neea, NEE, G3));
# QRE
NLL2[:,4] = nll(qre2, QRE, G2);
NLL3[:,4] = nll(qre3, QRE, G3);
NLLa[:,4] = vcat(nll(qrea, QRE, G2), nll(qrea, QRE, G3));
# LKr
NLL2[:,5] = nll(lkr2, LKr, G2);
NLL3[:,5] = nll(lkr3, LKr, G3);
NLLa[:,5] = vcat(nll(lkra, LKr, G2), nll(lkra, LKr, G3));
# CH1
NLL2[:,6] = nll(ch12, CH1, G2);
NLL3[:,6] = nll(ch13, CH1, G3);
NLLa[:,6] = vcat(nll(ch1a, CH1, G2), nll(ch1a, CH1, G3));
# NI
NLL2[:,7] = nll(ni2, NI, G2);
NLL3[:,7] = nll(ni3, NI, G3);
NLLa[:,7] = vcat(nll(nia, NI, G2), nll(nia, NI, G3));
# QLK
NLL2[:,8] = nll(qlk2, QLK, G2);
NLL3[:,8] = nll(qlk3, QLK, G3);
NLLa[:,8] = vcat(nll(qlka, QLK, G2), nll(qlka, QLK, G3));
# QCH1
NLL2[:,9] = nll(qch12, QCH1, G2);
NLL3[:,9] = nll(qch13, QCH1, G3);
NLLa[:,9] = vcat(nll(qch1a, QCH1, G2), nll(qch1a, QCH1, G3));
# QPLK
NLL2[:,10] = nll(qplk2, QPLK, G2);
NLL3[:,10] = nll(qplk3, QPLK, G3);
NLLa[:,10] = vcat(nll(qplka, QPLK, G2), nll(qplka, QPLK, G3));

Z2 = Vuong(NLL2, d)
Z3 = Vuong(NLL3, d)
Za = Vuong(NLLa, d)

## Generate Figure 5: Normalized loss values in game 1 for NLL, MSD, and MAE
using Plots

normalize(v::Vector) = (v .- minimum(v)) ./ (maximum(v) - minimum(v));
# game 1: 18-12
p18 = 0.61:0.01:0.99;
p12 = 1 .- p18;
p = hcat(p18, p12);
n = length(p18);
f = [64, 16];

NLL1 = normalize([nloglike(f, p[i,:]) for i=1:n]);
MSD1 = normalize([msd(f, p[i,:]) for i=1:n]);
MAE1 = normalize([mae(f, p[i,:]) for i=1:n]);

plot(p18, [NLL1, MSD1, MAE1], lw=2.5, label=["NLL" "MSD" "MAE"],
        titlefontsize=11, xlabelfontsize=10)
xlabel!("Estimated probability of choice 18");
title!("Normalized loss values in game 1 (18-12)")
png("loss")


# game 4: 18-12-12-12-12
p18 = 0.3:0.01:0.9;
p12 = @. (1 - p18)/4;
p = hcat(p18, p12, p12, p12, p12);
n = length(p18);
f = [48, 8, 8, 8, 8];

NLL4 = normalize([nloglike(f, p[i,:]) for i=1:n]);
MSD4 = normalize([msd(f, p[i,:]) for i=1:n]);
MAE4 = normalize([mae(f, p[i,:]) for i=1:n]);

plot(p18, [NLL4, MSD4, MAE4], lw=2.5, label=["NLL" "MSD" "MAE"],
        titlefontsize=11, xlabelfontsize=10)
xlabel!("Estimated probability of choice 18");
title!("Normalized loss values in game 4 (18-12-12-12-12)")
png("loss")

# game 12: 18-12-9
p18 = 0.3:0.01:0.9;
p9 = @. (1 - p18)/4;
p12 = 3p9;

p = hcat(p18, p12, p9);
n = length(p18);
f = [48, 25, 8];

NLL12 = normalize([nloglike(f, p[i,:]) for i=1:n]);
MSD12 = normalize([msd(f, p[i,:]) for i=1:n]);
MAE12 = normalize([mae(f, p[i,:]) for i=1:n]);

plot(p12, [NLL12, MSD12, MAE12], lw=2)
