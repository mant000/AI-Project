def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d / (1 + p) * log1p((progress + stagnation) / (max_d + mean_d)) ** 0.5
