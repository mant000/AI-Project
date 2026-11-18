def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return d / (1 + p) * (1 + 0.1 * (mean_d / max_d)) * (1 + 0.1 * (progress / stagnation)) * (1 + 0.1 * (1 - best_ratio))
