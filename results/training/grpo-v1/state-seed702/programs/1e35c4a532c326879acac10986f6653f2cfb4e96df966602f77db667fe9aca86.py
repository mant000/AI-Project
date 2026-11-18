def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (max_d + 0.25) / (1 + p)
