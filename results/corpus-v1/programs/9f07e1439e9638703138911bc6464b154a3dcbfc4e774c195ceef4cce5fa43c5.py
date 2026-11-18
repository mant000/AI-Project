def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d * exp(-0.05 * p)
