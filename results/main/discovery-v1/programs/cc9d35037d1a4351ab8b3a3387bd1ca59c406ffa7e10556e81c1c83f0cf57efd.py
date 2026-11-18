def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (d + p ** 2) / (1 + p)
