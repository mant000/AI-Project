def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.5 * d / (1 + p) + 0.5 * (max_d - mean_d) / (p + 1e-06)
