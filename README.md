#  <p align="center">SyMANTIC: An Efficient Symbolic Regression Method for Interpretable and Parsimonious Model Discovery in Science and Beyond

![toc](https://i.ibb.co/4Nmvj3B/symantic-toc.jpg)


## Quick Start 


 A standard use case of the SyMANTIC code can be compactly written as follows:

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



### Citation
    Coming soon
