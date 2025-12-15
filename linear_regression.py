import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class lin_reg:
    def __init__ (self):
        pass

    def calc(self,df):
        x = (df.drop(columns=['y']))
        y = df['y']

        pipe = Pipeline([
            ("scaler",StandardScaler()),
            ('lr',LinearRegression())
        ])

        pipe.fit(x,y)

        x_vals = np.linspace(
            x['default'].min(),
            x['default'].max(),
            200
        )

        x_line = pd.DataFrame(
            np.tile(x.mean().values, (200,1)),
            columns=x.columns
        )
        x_line['default'] = x_vals

        y_line = pipe.predict(x_line)

        return x_vals, y_line
    
lin_reg_calc=lin_reg()

