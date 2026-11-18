def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return log1p(0.25 * d / mean_d) / (1 + p) ** 1.5
