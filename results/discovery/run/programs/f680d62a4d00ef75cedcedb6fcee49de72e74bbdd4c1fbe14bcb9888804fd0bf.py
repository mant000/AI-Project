def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return 0.5 * d / (1 + p) + 0.5 * d / (max_d / mean_d) * (1 - 0.25 * progress / log1p(10))
