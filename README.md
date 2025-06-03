## Behavioral Model Selection in Symmetric Normal-Form Games

This repository contains the data and code necessary to reproduce the results presented in the paper *Behavioral Model Selection in Symmetric Normal-Form Games* (Emirahmetoglu, Goksel, Gurdal, 2025). The paper can be downloaded [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5204145).

The [data](/data) folder contains the experimental results: 
* [G2](/data/G2.csv) provides the results for the two-player games.
* [G3](/data/G3.csv) provides the results for the three-player games.

The [src](/src) folder includes the following Julia programs:
* [games.jl](/src/games.jl) defines two- and three-player games (set of actions $A$, payoff function $u$, derivative of the payoff function $du$) and provides the observed choice frequencies $f$ for each game. 
* [models.jl](/src/models.jl) defines the behavioral models.
* [Nash.jl](/src/Nash.jl) computes the symmetric Nash equilibrium for each game. 
* [estimation.jl](/src/estimation.jl) defines loss functions, statistical (in-sample) fit and out-of-sample estimation functions along with Vuong's test.
* [main.jl](/src/main.jl) is the main file to be used. Statistical fits, out-of-sample predictions, Vuong's test are performed here. 

### Directions to reproduce the estimation results

(1) Make sure you have the [src](/src) folder in your current directory. To change your current directory, you can adapt and use the following command in the Julia REPL:

    cd("/Users/Desktop/games/src")

(2) Create two subfolders in the [src](/src) folder, named "statistical_fit" and "out_of_sample". When you run the estimations, the results will be saved under these folders.

(3) Make sure to install the following packages in Julia: `BlackBoxOptim`, `Combinatorics`, `Distributions`, `GameTheory`, `LinearAlgebra`, `NNlib`, `Plots`, `Printf`, and `Statistics`. For example, to install the `LinearAlgebra` package, run the following command in the Julia REPL:

    import Pkg; Pkg.add("LinearAlgebra")

(4) To reproduce the results, consider the [main.jl](/src/main.jl) file. When you run the following commands in the Julia REPL, all the necessary modules (`Estimation`, `Games`, and `Models`) will be imported, and you will be ready to execute any estimations:

    include("estimation.jl");  using .Estimation

(5) The function `fit` is for statistical fits, and `train` is for out-of-sample estimations. You need to pass the model, loss function, and games to these functions. 
* The models are `NEE`, `QRE`, `LKr`, `LK1`, `CHr`, `CH1`, `NI`, `QLK`, `QCHr`, `QCH1`, `HNIr`, `HNI1`, `GCHr`, `GCH1`, `LMr`, `LM1`, `QPLK`. 
* The loss functions are `NLL`, `MSD`, `MAE`.
* The games are `G2` (two-player), `G3` (three-player), and `G2` & `G3` (all games). 

To find the statistical fit of the `model` in the games `G` using the `loss` function, you simply run

    fit(model, loss, G)

To find the out-of-sample estimation of the `model` in the games `G` using the `loss` function, you simply run

    train(model, loss, G)

The results will be saved in the subfolders "statistical_fit" and "out_of_sample" you created under the [src](/src) folder.

For example, to find the statistical fit of `QRE` in two-player games using `NLL`, run 

    fit(QRE, NLL, G2)

To find the statistical fit of `CHr` in three-player games using `MSD`, run 

    fit(CHr, MSD, G3)

To find the statistical fit of `NI` in all games using `MAE`, run

    fit(NI, MAE, G2, G3)

To find the out-of-sample estimations of `QLK` in two-player games using `NLL`, run

    train(QLK, NLL, G2)

To find the out-of-sample estimations of `HNI1` in three-player games using `MSD`, run

    train(HNI1, MSD, G3)

To find the out-of-sample estimations of `QPLK` in all games using `MAE`, run

    train(QPLK, MAE, G2, G3)

