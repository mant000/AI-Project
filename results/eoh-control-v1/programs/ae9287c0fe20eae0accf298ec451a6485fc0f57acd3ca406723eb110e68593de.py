def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d * (1 - p) / (sqrt(1 + p) * (1 + exp(-lam * progress)) * (1 + stagnation))
