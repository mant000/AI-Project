def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.6 * d / (1 + p) + 0.3 * (mean_d - d) / (1 + p) + 0.1 * (max_d - d) / (1 + p)
