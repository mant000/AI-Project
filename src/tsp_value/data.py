"""Parse author rows and keep distance conventions explicit."""
import hashlib
import struct
import numpy as np


def coordinate_hash(coords, float32=False):
    """Hash unordered coordinates, optionally after float32 quantization."""
    coords = np.asarray(coords, dtype=np.float64)
    if float32:
        coords = coords.astype(np.float32).astype(np.float64)
    return hashlib.sha256(b''.join(struct.pack('<dd', *p) for p in sorted(map(tuple, coords)))).hexdigest()


def parse_author(raw, n):
    """Return float64 coordinates and a checked zero-based closed reference tour."""
    tokens = raw.decode('utf-8').split()
    k = tokens.index('output')
    if k != 2*n:
        raise ValueError('Coordinate count does not match dimension')
    coords = np.asarray(tokens[:k], dtype=np.float64).reshape(n, 2)
    tour = np.asarray(tokens[k+1:], dtype=np.int64)-1
    if not np.isfinite(coords).all() or (coords < 0).any() or (coords > 1).any():
        raise ValueError('Invalid unit-square coordinates')
    if len(tour) != n+1 or tour[0] != tour[-1]:
        raise ValueError('Reference tour must be closed')
    validate_tour(tour[:-1], n)
    return coords, tour[:-1]


def validate_tour(tour, n):
    """Reject a route that is not exactly a permutation of all cities."""
    if len(tour) != n or not np.array_equal(np.sort(tour), np.arange(n)):
        raise ValueError('Tour is not a permutation')


def distances(coords, convention='continuous_euclidean'):
    """Continuous Euclidean for Joshi/INViT; TSPLIB EUC_2D rounds half up."""
    x = np.asarray(coords, dtype=np.float64)
    d = np.sqrt(((x[:, None, :] - x[None, :, :])**2).sum(axis=2))
    if convention == 'tsplib_euc_2d':
        d = np.floor(d+0.5)
    elif convention != 'continuous_euclidean':
        raise ValueError(convention)
    return d


def length(tour, d):
    """Closed-tour cost, always recomputed against the requested matrix."""
    return float(d[tour, np.roll(tour, -1)].sum())
