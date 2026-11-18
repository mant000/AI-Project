def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d / (1 + p + 0.1 * stagnation + 0.1 * mean_d)
