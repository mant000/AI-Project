def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return (d - 0.075 * max_d) ** 0.95 / (1 + p)
