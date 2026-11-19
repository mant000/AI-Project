def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.6 * d / (1 + p) + 0.4 * d / (1 + p) ** 2
