"""Fetch MNIST once and save a fixed subsample to data/mnist.csv."""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SEED = 42
N_SAMPLES = 10000

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X, _, y, _ = train_test_split(
    X, y, train_size=N_SAMPLES, random_state=SEED, stratify=y
)

df = pd.DataFrame(X.astype(np.uint8))
df.columns = [f"p{i}" for i in range(X.shape[1])]
df.insert(0, "label", y)
df.to_csv("data/mnist.csv", index=False)
print(f"wrote data/mnist.csv with {len(df)} rows")
