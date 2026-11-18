def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (1 - clip(progress / 2, 0, 1)) * d / (1 + p) + clip(progress / 2, 0, 1) * (d - mean_d) / (1 + p) ** 1
