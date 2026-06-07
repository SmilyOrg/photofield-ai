"""
Minimal pure-numpy SimilarityTransform replacement for skimage.transform.SimilarityTransform.
Injected at Docker build time to eliminate the scipy+scikit-image dependency chain (~142 MB).

Supports the full API used by uniface/face_utils.py:
  - Constructor kwargs: scale, rotation, translation, matrix
  - SimilarityTransform.from_estimate(src, dst)  -- class method (skimage >= 0.26)
  - instance.estimate(src, dst)                   -- deprecated instance method
  - t1 + t2                                       -- compose (t1 applied first)
  - instance.params                               -- 3x3 homogeneous matrix
"""

import numpy as np


class SimilarityTransform:
    def __init__(self, scale=None, rotation=None, translation=None, matrix=None):
        if matrix is not None:
            self.params = np.asarray(matrix, dtype=float)
            return
        s = float(scale) if scale is not None else 1.0
        r = float(rotation) if rotation is not None else 0.0
        tx, ty = (float(translation[0]), float(translation[1])) if translation is not None else (0.0, 0.0)
        c, ss = np.cos(r), np.sin(r)
        self.params = np.array(
            [[s * c, -s * ss, tx],
             [s * ss,  s * c, ty],
             [0.0,     0.0,   1.0]]
        )

    def __add__(self, other):
        result = SimilarityTransform()
        result.params = other.params @ self.params
        return result

    def estimate(self, src, dst):
        M = _umeyama(np.asarray(src, dtype=float), np.asarray(dst, dtype=float))
        if M is None:
            return False
        self.params = M
        return True

    @classmethod
    def from_estimate(cls, src, dst):
        t = cls()
        t.estimate(src, dst)
        return t


def _umeyama(src, dst):
    """Umeyama similarity-transform estimation (scale + rotation + translation)."""
    n, m = src.shape
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    src_var = np.mean(np.sum(src_c ** 2, axis=1))
    if src_var < 1e-10:
        return None
    cov = (dst_c.T @ src_c) / n
    U, d, Vt = np.linalg.svd(cov)
    D = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1
    c = (d * D.diagonal()).sum() / src_var
    R = U @ D @ Vt
    M = np.eye(m + 1)
    M[:m, :m] = c * R
    M[:m, m] = dst_mean - c * R @ src_mean
    return M
