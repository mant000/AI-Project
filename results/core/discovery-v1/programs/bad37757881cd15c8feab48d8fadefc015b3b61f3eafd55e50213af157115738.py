def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d / (1 + p) - lam * p + 0.01 * (progress + stagnation)
