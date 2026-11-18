def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (d - 0.2 * max_d - 0.1 * (max_d - mean_d)) / (1 + p)
