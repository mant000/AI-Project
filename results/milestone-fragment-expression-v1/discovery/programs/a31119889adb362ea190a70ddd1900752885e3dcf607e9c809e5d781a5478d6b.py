def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.25 * d / (1 + p) / (1 + 0.1 * progress) ** 2
