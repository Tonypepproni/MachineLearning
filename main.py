import scrubber 
import linear_regression
from k_neighbor import k_nei
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('bank-full.csv',sep=';')
print(df.head())
df=scrubber.scrubber.clean(df)
print(df.head())

x_vals, y_line = linear_regression.lin_reg_calc.calc(df)

plt.plot(x_vals,y_line)
plt.show()