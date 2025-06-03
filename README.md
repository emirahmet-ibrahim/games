## Behavioral Model Selection in Symmetric Normal-Form Games

This repository contains the data and code necessary to reproduce the results presented in the paper *Behavioral Model Selection in Symmetric Normal-Form Games* (Emirahmetoglu, Goksel, Gurdal, 2025). The paper can be downloaded [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5204145).

The [src](/src) folder includes the following Julia programs:
* [games.jl](/src/games.jl) defines two- and three-player games (set of actions $A$, payoff function $u$, derivative of the payoff function $du$) and provides the observed choice frequencies $f$ for each game. 
* [models.jl](/src/models.jl) defines the behavioral models.
* [Nash.jl](/src/Nash.jl) computes the symmetric Nash equilibrium for each game. 
* [estimation.jl](/src/estimation.jl) defines loss functions, statistical (in-sample) fit and out-of-sample estimation functions along with Vuong's test.
* [main.jl](/src/main.jl) is the main file to be used. Statistical fits, out-of-sample predictions, Vuong's test are performed here. 
