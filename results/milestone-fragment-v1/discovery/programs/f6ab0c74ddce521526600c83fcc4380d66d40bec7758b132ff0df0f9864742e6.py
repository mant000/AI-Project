def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d / (1 + p + 0.1 * stagnation) * (1 - 0.05 * (max_d - mean_d) / max_d)
