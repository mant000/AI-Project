def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d + p * (max_d - mean_d) ** 2 * (1 + progress) * (1 + stagnation) * (best_ratio - 1) / (1 + p)
