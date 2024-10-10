#  <p align="center">SyMANTIC: An Efficient Symbolic Regression Method for Interpretable and Parsimonious Model Discovery in Science and Beyond

![](https://i.ibb.co/4Nmvj3B/symantic-toc.jpg)

SyMANTIC is a novel SR algorithm that efficiently identifies low-dimensional features set from an enormous set of candidates through a unique combination of mutual information-based feature selection, adaptive feature expansion, and recursively applied $\ell_0$-based sparse regression. Additionally, it employs an information-theoretic measure to produce a set of Pareto-optimal equations, each offering the best accuracy for a given complexity. This open-source implementation of SyMANTIC is built on the PyTorch ecosystem.

## Quick Start 


Install SyMANTIC and dependancies
```bash
pip install symantic
```

Import your data and use the following code to fit a SyMANTIC model and analyze the Pareto front
```python 
# import SyMANTIC model class along with other useful packages
from symantic import SymanticModel
import numpy as np
import pandas as pd
# create dataframe composed of targets "y" and primary features "X"
data = np.column_stack((y, X))
df = pd.DataFrame(data)
# create model object to contruct full Pareto using default parameters
model = SymanticModel(df=df, #defines the dataframe,
                      operators = ['+','-','*','/','exp','sin','cos'], #defines the set of operators for feature engineering
                      n_epxansion = None, (default) # Defines the number of feature expansions, if a value is provided then
                      n_term = None, #defines the sparsity that needs to be considered for building models
                      sis_features = 20, (default) # defines the number of features to be screened from the expanded feature space
                      dimensionality = ['u1','u2','u3'], #Defines the units of the feature variables in string representation which later converted into sympy format to do                                                            #the meaningful feature construction.
                      relational_units = [(symbols('u1')*symbols('u2'),symbols('u3)], #Defines the list of tuples where each tuple represents the relational transformation.
                      output_dim = (symbols('u1')*symbols('u1')), #Defines the units of the target variable which helps in narrowing down the space for Regularization.
                      initial_screening = ["mi" or "spearman", quantile value], #Defines the feature screening option for high dimensional and 1-quantile_value defines
                      metrics = [RMSE, $R^2$], #defines the values of RMSE and $R^2$ that are used to do the adaptive expansions and number of terms
                      disp = True or False #defines whether to print the statements of progress.
                      )
# run SyMANTIC algorithm to fit model and return dictionary "res" and "full_pareto" frontier
res,full_pareto = model.fit()
# generate plot of Pareto front obtained during the fit process
model.plot_pareto_front()
# extract symbolic model at utopia point and relevant metrics
model = res['utopia']['expression']
rmse = res['utopia']['rmse']
r2 = res['utopia']['r2']
complexity = res['utopia']['complexity']
```


Examples of SyMANTIC can be found in Examples folder and in the Colab Notebook [SyMANTIC Examples](https://colab.research.google.com/drive/1dBc2QJeEjW0T8iobFU8F54Y25pxR7isG#scrollTo=60564135 )



### Citation
    Coming soon
