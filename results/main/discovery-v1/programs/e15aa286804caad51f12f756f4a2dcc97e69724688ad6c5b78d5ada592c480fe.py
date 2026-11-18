def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d / (1 + 0.2 * p)
