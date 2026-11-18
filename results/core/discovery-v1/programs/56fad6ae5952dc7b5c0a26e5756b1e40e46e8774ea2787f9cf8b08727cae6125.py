def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d / (1 + p) - log1p(progress + stagnation) / sqrt(max_d)
