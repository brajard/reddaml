

# Run the experiments of the Lorenz 2 scale model

## Configuration
You can change the [config/paths.yml](config/paths.yml) to set the directory where to download the data
(TODO: estimate the space needed)

## Reproduce the figures of the paper

## Run the model from the scripts


### Run only the reference simulation

Training true simulation:\
```python simul.py --paths config/paths.yml --params config/sens_train.yml --model config/model_true.yml```

Training truncated simulation:\
```python simul.py --paths config/paths.yml --params config/sens_train.yml --model config/model_trunc.yml```

Testing true simulation:\
```python simul.py --paths config/paths.yml --params config/sens_test.yml --model config/model_true.yml```

Testing truncated simulation:\
```python simul.py --paths config/paths.yml --params config/sens_test.yml --model config/model_trunc.yml```

### Run all the simulations


## Run the model from the notebooks
