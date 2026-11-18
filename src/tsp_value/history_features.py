"""Current-configuration encoding with explicit penalty/incumbent/counter memory removed."""
import numpy as np
from .value_model import state_features


def history_features(state, edges, arm):
    """Keep the input width/architecture fixed; return no fitted or future information."""
    if arm == 'full': return state_features(state, edges)
    if arm != 'no_history': raise ValueError(arm)
    current = dict(state)
    current.update(penalties=np.zeros_like(state['penalties']), best_tour=np.asarray(state['tour']),
        best_length=state['current_length'], penalty_events=0, stagnation_sweeps=0, delta_evaluations=0)
    return state_features(current, edges)
