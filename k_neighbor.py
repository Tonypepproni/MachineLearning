import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class k_nei:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.pipe = None

    def fit(self, df):
        x = df.drop(columns=['y'])
        y = df['y']
        self.pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('neigh', KNeighborsClassifier(n_neighbors=self.n_neighbors))
        ])
        self.pipe.fit(x, y)
        self.x_mean = x.mean()  # store mean of features for plotting

    def predict_line(self, df, feature='default', num_points=200):
        # create a grid along the chosen feature
        x_vals = np.linspace(df[feature].min(), df[feature].max(), num_points)
        x_line = pd.DataFrame(np.tile(self.x_mean.values, (num_points, 1)), columns=df.drop(columns=['y']).columns)
        x_line[feature] = x_vals
        y_line = self.pipe.predict(x_line)
        return x_vals, y_line

knn_model = k_nei(n_neighbors=5)
        