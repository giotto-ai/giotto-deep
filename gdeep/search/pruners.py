import optuna
from optuna.pruners import BasePruner

class VariationPruner(BasePruner):
    """Prunes trial if the number of scores is less than ´possible_values´.
    """
    def __init__(self, warmup_steps, ):
        self._warmup_steps = warmup_steps
        self._possible_values = possible_values

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        # Get the latest score reported from this trial
        step = trial.last_step

        if step:  # trial.last_step == None when no scores have been reported yet
            scores_set = set(trial.intermediate_values)

            # Get scores from previous steps


            # Prune if this trial at this step has a lower value than all completed trials
            # at the same step. Note that steps will begin numbering at 0 in the objective
            # function definition below.
            if step >= self._warmup_steps and len(scores_set) > self._possible_values:
                print(f"prune() True: Trial {trial.number}, Step {step}, Score {trial.intermediate_values[-1]}")
                return True

        return False