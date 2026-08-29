#!/usr/bin/env bash
set -e

python src/prepare_data.py
wc -l data/mnist.csv

dvc add data/mnist.csv
dvc push

git add src/prepare_data.py src/train.py data/mnist.csv.dvc data/.gitignore
git commit -m "Partner A: MNIST MLP training code with DVC-versioned dataset v1"

python src/train.py --seed 42 --register

git rev-parse HEAD
