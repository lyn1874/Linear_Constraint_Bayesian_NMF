### Linearly constraint Bayesian Matrix Factorization for Blind Source Separation

This repository provides the implementation for the paper [**Linearly constraint Bayesian Matrix Factorization for Blind Source Separation** (Mikkel N. Schmidt)](https://proceedings.neurips.cc/paper/2009/hash/371bce7dc83817b7893bcdeed13799b5-Abstract.html). Majority of the code are translated from the [Matlab implementation](http://mikkelschmidt.dk/code.html) that is provided by Mikkel N. Schmidt



#### Installation and preparation 
1. Clone this repo:

   ```bash
   git clone https://github.com/lyn1874/Linear_Constraint_Bayesian_NMF.git
   cd Linear_Constraint_Bayesian_NMF
   ```

2. Requirement:
```
python3/3.7.7  
matplotlib/3.2.1-python-3.7.7  
scipy/1.4.1-python-3.7.7  
pandas/1.0.3-python-3.7.7
```


#### Train the model
2. Train the model:
    ```bash
    ./run.sh dataset N mu_prior infinity
	Args:
		dataset: mnist 
		N: number of components, int
		mu_prior: the mean of the prior distribution for component matrix A and mixing coeffients B
		infinity: bool variable. If True, then the variance of the prior distribution for A and B are infinitely large. 	
    ```
    
#### Citation
If you use this code for your research, please cite the paper:
```
@inproceedings{NIPS2009_371bce7d,
 author = {Schmidt, Mikkel},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {Y. Bengio and D. Schuurmans and J. Lafferty and C. Williams and A. Culotta},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Linearly constrained Bayesian matrix factorization for blind source separation},
 url = {https://proceedings.neurips.cc/paper/2009/file/371bce7dc83817b7893bcdeed13799b5-Paper.pdf},
 volume = {22},
 year = {2009}
}
```