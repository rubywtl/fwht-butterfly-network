import math
import numpy as np
import matplotlib.pyplot as plt

# NAIVE SOLUTION (strictly power of 2)
def hadamard_matrix(n):
    assert (n & (n - 1)) == 0, "n must be power of 2"
    H = np.array([[1]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]]) / np.sqrt(2)
    return H

def naive_fwht(a: np.ndarray) -> np.ndarray:
    assert math.log2(len(a)).is_integer(), "must be the power of 2"
    H = hadamard_matrix(len(a))
    return np.round(np.dot(H, a))

# NAIVE SOLUTION (not restricted to size pow of 2)


# FWHT BUTTERFLY NETWORK (strictly power of 2)
def fwht(a) -> None:
    assert math.log2(len(a)).is_integer(), "must be the power of 2"
    h = 1
    while h < len(a):
        # perform FWHT
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]

                # butterfly network core computation
                a[j] = x + y
                a[j + h] = x - y

        # normalize
        a /= math.sqrt(2)
        h *= 2


# FWHT BUTTERFLY NETWORK (not restricted to size pow of 2)
def fwht_unfixed(a: np.ndarray) -> np.ndarray:
    orig_len = len(a)
    next_pow2 = 1 << (orig_len - 1).bit_length()  # nearest power of 2 >= orig_len

    if orig_len != next_pow2:
        padded = np.zeros(next_pow2, dtype=a.dtype)
        padded[:orig_len] = a
    else:
        padded = a.copy()

    fwht(padded)
    return padded
    



