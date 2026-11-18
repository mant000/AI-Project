def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.7 * d / (1 + p) + 0.3 * (1 - mean_d / max_d)
