def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.4 * d / (1 + p) + 0.6 * (mean_d - max_d) / (p + 1e-06)
