# UMAB-SC
My master thesis in Machine Learning

## TODO
- ~~each experiment repeated 5 times~~
- ~~save experiments on file separating the execution of the policies from computation and plotting of regrets~~
- ~~add the history of switching costs to the choose_arm method~~
- ~~add stochatic switching costs~~
- add other policies
  - ~~KL-UCB~~
  - ~~UCYCLE~~
  - ~~Thopmson Sampling~~
  - ~~Thompson Sampling in epochs~~
  - ~~UCB1 that takes into account switching costs~~
  - ~~UCBV~~
  - ~~MOSS~~
- ~~compute theoretical bounds of each policy~~ (mancano UCBV, TS2 e UCB1SC) (problema con UCYCLE: numeri troppo grandi)
- ~~compute variance of the regrets~~
- ~~plot variance and theoretical bound~~
- ~~adjust file saving procedure by separating in subfolders different policies~~
- ~~Parameter tuning:
  - tune the alpha parameter of UCB2 and TS2
  - tune the delta parameter of UCYCLE
  - tune the parameters of UCBV~~
- Add budget regret
- Save policies and configuration separatly so to reduce memory usage
