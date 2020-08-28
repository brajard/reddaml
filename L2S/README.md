

# Run the experiments of the Lorenz 2 scale model

## Configuration
You can change the [config/paths.yml](config/paths.yml) to set the directory where to download the data
(TODO: estimate the space needed)

## Reproduce the figures of the paper

## Run the model from the scripts


### Run only the reference simulation


### Run all the simulations

Training true simulation:\
```python simul.py --paths config/paths.yml --params config/sens_train.yml --model config/model_true.yml```

Training truncated simulation:\
```python simul.py --paths config/paths.yml --params config/sens_train.yml --model config/model_trunc.yml```

Testing true simulation:\
```python simul.py --paths config/paths.yml --params config/sens_test.yml --model config/model_true.yml```

Testing truncated simulation:\
```python simul.py --paths config/paths.yml --params config/sens_test.yml --model config/model_trunc.yml```

Compute the training set with noisy/sparse observation (using DA):\
```python compute_trainingset.py --paths config/paths.yml --params config/sens_train.yml --model config/model_trunc.yml```

Compute the training set with perfect observation (no DA):\
```python compute_trainingset.py --paths config/paths.yml --params config/sens_train_po.yml --model config/model_trunc.yml```

Train the NN with noisy/sparse observation:\
```python train.py --paths config/paths.yml --params config/sens_train.yml```

Train the NN with perfect observation:\
```python train.py --paths config/paths.yml --params config/sens_train_po.yml```

Testing hybrid simulation:\
```python simul.py --paths config/paths.yml --params config/sens_test.yml --model config/model_hybrid.yml```


Testing hybrid simulation with perfect observations:\
```python simul.py --paths config/paths.yml --params config/sens_test_po.yml --model config/model_hybrid.yml```
## Run the model from the notebooks
