import pickle
from typing import Tuple

import numpy as np
from kedro.pipeline import Pipeline, node
from sklearn.linear_model import LinearRegression


def preprocess_pipeline(data_num: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([n for n in range(data_num + 1)])
    y = np.asarray([2 * n + 5 for n in x])
    return np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)


def train_pipeline(x: np.ndarray, y: np.ndarray) -> None:
    model = LinearRegression()
    model.fit(x, y)
    filename = 'model.sav'
    pickle.dump(model, open(filename, 'wb'))


def create_pipelines(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocess_pipeline,
                inputs=["params:data_num"],
                outputs=["x", "y"],
            ),
            node(
                func=train_pipeline,
                inputs=["x", "y"],
                outputs=None,
            )
        ]
    )

