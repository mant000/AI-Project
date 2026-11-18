def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return maximum(d / mean_d - 0.25, 0.001) / (1 + p) ** 0.5
