def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.7 * d / (1 + p) + 0.2 * (mean_d - d) / (1 + p) + 0.05 * (max_d - d) / (1 + p) + 0.05 * (best_ratio - 1) / (1 + p) + 0.05 * (progress / (1 + stagnation)) / (1 + p)
