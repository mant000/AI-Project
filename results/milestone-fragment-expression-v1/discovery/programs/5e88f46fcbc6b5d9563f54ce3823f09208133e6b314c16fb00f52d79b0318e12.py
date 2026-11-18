def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.8 * d / (1 + p) + 0.15 * (mean_d - d) / (1 + p) + 0.05 * (max_d - d) / (1 + p)
