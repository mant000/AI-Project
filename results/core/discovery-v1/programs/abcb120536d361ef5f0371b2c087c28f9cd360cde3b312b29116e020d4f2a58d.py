def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d / (1 + p) * (mean_d / sqrt(mean_d ** 2 + max_d ** 2))
