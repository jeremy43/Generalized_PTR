# Generalized_PTR
This repository provides the implementation for the paper [Generalized PTR: User-Friendly Recipes for Data-Adaptive Algorithms with Differential Privacy](https://arxiv.org/pdf/2301.00301.pdf)

We apply the generalized PTR framework to solve an open problem from the Private
Aggregation of Teacher Ensembles (PATE) [Papernot et al., 2017, 2018] — privately publishing the
entire model through privately releasing data-dependent DP losses. Our algorithm makes use of the
smooth sensitivity framework [Nissim et al., 2007] and the Gaussian mechanism to construct a highprobability test of the data-dependent DP. 

## How to run?

`
python pate.py
`
