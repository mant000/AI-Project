def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (1 - 0.1 * p) * d / (1 + p)
