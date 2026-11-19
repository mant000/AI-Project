def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.5 * d / (1 + p) + 0.5 * (1 - mean_d / max_d)
