def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (d / mean_d - 0.5) / (1 + p) ** 1
