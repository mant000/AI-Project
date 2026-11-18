def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (d - 0.25 * lam * p) / (1 + p)
