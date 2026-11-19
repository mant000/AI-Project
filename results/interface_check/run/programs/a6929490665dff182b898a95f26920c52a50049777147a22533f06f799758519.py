def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (d / (1 + p)) ** 0.75 * (1 + (max_d - mean_d) / mean_d) ** 0.25
