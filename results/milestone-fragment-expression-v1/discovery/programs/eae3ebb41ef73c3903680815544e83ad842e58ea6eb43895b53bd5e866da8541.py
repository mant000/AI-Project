def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.7 * d / (1 + p) + 0.3 * d / ((1 + p) ** 2 + 0.01)
