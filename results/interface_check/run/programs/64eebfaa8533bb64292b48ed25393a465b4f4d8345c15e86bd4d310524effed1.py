def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (d - mean_d) / (max_d - mean_d) / (1 + p) - lam * p
