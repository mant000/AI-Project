def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.25 * d / (1 + p) + 0.75 * d / (1 + p) ** 2
