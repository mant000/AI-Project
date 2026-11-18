def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (d - 4 * lam * p) / (1 + p)
