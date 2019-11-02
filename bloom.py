import numpy as np
import math
from typing import Set, Union, Iterable

from shish import Hash

class BloomFilter:

    def __init__(self, M: int, d: int, s: Set[int]=None):
        """
        Probabilistic structure introduced in [1],
        used to test if a value belongs to a set or not.
        The test is guaranteed to be positive if the value has been seen,
        but it can return false positives with some probability that
        depends on M, d, and the number of elements added to the set.

        Args:
            M (int): the numer of bits
            d (int): the number of hash functions
            s (set): a set of values to initialize the structure with

        [1] H. Bloom. Space/Time Trade-offs in Hash Coding with Allowable Errors
        """

        self.M, self.d = M, d
        self.X = np.zeros(self.M, dtype=bool)
        self.hash = Hash(self.d, self.M)
        if s:
            self.add(s)

    def _add(self, x: int):
        assert x >= 0, "Only non-negative integers are supported"
        self.X[self.hash(x)] = 1 

    def add(self, xs: Union[int, Iterable[int]]):
        if isinstance(xs, int):
            xs = [xs]
        for x in xs:
            self._add(x)

    def __contains__(self, x: int):
        return np.all(self.X[self.hash(x)])

    def stats(self, N: int):
        """
        Compute and print the probability of false positives of
        the structure with a given number of elements.
        """
        prob = (1 - math.exp(- self.d * N / self.M))**self.d
        print(f"Given {N} inserted elements, "
              f"the probability of false positives is ~{100*prob:.2f}%.")

    @staticmethod
    def optimal_size(p: float, N: int):
        """Optimal M and d for N inserted elements and a desired false positive rate p"""
    
        M = math.ceil(- N * math.log2(p) / math.log(2))
        d = math.ceil(- math.log2(p))
        return M, d
