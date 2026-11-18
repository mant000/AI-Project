def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d / (1 + p) * (1 + exp(-lam * (log1p(progress) + log1p(stagnation)))) * sqrt(max_d)
