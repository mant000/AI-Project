def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d / (1 + p) * exp(-sqrt(log1p(progress * mean_d / max_d)))
