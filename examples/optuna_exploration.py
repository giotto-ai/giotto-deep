# %%
# Example: Optimization of a quadratic funtion
import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -1, 1)
    y = trial.suggest_uniform('y', -1, 1)
    return x**2 + y**2

sampler = optuna.samplers.TPESampler(multivariate=False)  # use multivariate TPE sampler
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)

# %%
x = [study.trials[i].params['x'] for i in range(0, len(study.trials))]
y = [study.trials[i].params['y'] for i in range(0, len(study.trials))]

import matplotlib.pyplot as plt

plt.scatter(x, y)
# %%
