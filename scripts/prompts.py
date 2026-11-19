"""The fixed system and user instructions used in both response interfaces."""
SYSTEM='You design a pure numeric edge-priority rule for guided local search for Euclidean TSP. Obey the supplied interface. Return only Python code. No imports, comments, prose, assignments or extra functions.'

TASK='''Improve the given parent using real endpoint results on fixed discovery instances.
The solver penalizes one maximum-priority edge of the CURRENT tour then performs budget-exact best-improvement 2-opt. Quality always uses original Euclidean route length. Lower gap-to-reference is better; reference routes are not certified optima. Only the priority rule may change. Search uses a fixed work cap; do not infer improvement from a predicted value alone.
Return exactly:
def priority(d, p, mean_d, max_d, lam, progress, stagnation, best_ratio):
    return ONE_ARITHMETIC_EXPRESSION
d is the array of current edge lengths, p their nonnegative penalty counts. mean_d/max_d are current-tour distance summaries, lam is the frozen GLS penalty weight, progress=log1p(number of penalty events), stagnation=log1p(stagnation sweeps), best_ratio=best/current route length. Arithmetic + - * / ** and bare sqrt, log1p, exp, minimum, maximum, abs, clip are allowed. No np prefix or attributes, indexing, conditionals, loops or assignments. Max 96 AST nodes, finite constants magnitude <=100, literal powers magnitude <=4. Use numerically safe expressions. Scalar constants broadcast. Standard GLS is d/(1+p). Prefer a concise meaningful change to the parent. Every output, including invalid or duplicate code, consumes the budget.
'''
