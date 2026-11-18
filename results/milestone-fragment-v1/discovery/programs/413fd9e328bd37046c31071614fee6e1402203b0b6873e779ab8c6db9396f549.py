def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (d - 0.2 * max_d) / (1 + p)
