def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (d - max_d * (0.2 - 0.1 * (mean_d / max_d))) / (1 + p)
