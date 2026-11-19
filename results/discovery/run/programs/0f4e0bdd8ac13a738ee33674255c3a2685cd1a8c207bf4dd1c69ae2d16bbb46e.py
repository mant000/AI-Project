def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.8 * d / (1 + p) + 0.1 * (mean_d - d) / (1 + p) + 0.1 * (max_d - d) / (1 + p) + 0.1 * (best_ratio - 1) / (1 + p) + 0.05 * (progress / (1 + stagnation)) / (1 + p)
