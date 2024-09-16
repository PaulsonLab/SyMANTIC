#  <p align="center">SyMANTIC: An Efficient Symbolic Regression Method for Interpretable and Parsimonious Model Discovery in Science and Beyond

![](https://i.ibb.co/4Nmvj3B/symantic-toc.jpg)

SyMANTIC is a novel SR algorithm that efficiently identifies low-dimensional descriptors from an enormous set of candidates through a unique combination of mutual information-based feature selection, adaptive feature expansion, and recursively applied $\ell_0$-based sparse regression. Additionally, it employs an information-theoretic measure to produce a set of Pareto-optimal equations, each offering the best accuracy for a given complexity. This open-source implementation of SyMANTIC is built on the PyTorch ecosystem.

## Quick Start 


Install SyMANTIC and dependancies
```bash
pip install SyMANTIC
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
model = SymanticModel(data=df, pareto=True)
# run SyMANTIC algorithm to fit model and return dictionary "res"
res = model.fit()
# generate plot of Pareto front obtained during the fit process
model.plot_Pareto_front()
# extract symbolic model at utopia point and relevant metrics
model = res['utopia']['expression']
rmse = res['utopia']['rmse']
r2 = res['utopia']['r2']
complexity = res['utopia']['complexity']
```


Examples of SyMANTIC can be found in Examples folder and in the Colab Notebook [SyMANTIC Examples](https://colab.research.google.com/drive/1dBc2QJeEjW0T8iobFU8F54Y25pxR7isG#scrollTo=60564135 )



### Citation
    Coming soon
