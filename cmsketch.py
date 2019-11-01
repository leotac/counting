import numpy as np
import math

class CountMinSketch:

    def __init__(self, ϵ: int, δ: int, noise_correction: str=None):
        """
        Keeps an approximate count of each distinct value seen in a data stream [1,2].
        This implementation only works for *positive* counts.
        
        Args:
            ϵ (float): controls the magnitude of the error
            δ (float): controls the fraction of estimates
                       that can exceed the error bound
            noise_correction: if not None, use a noise correction method from [3]
                              Available options: 'mean' and 'median'.

        [1] G. Cormode, S. Muthukrishnan. An Improved Data Stream Summary:
                                          The Count-Min Sketch and its Applications
        [2] G. Cormode, S. Muthukrishnan. Approximating Data with the Count-Min Data Structure
        [3] F. Deng, D. Rafiei. New Estimation Algorithms for Streaming Data: Count-min Can Do More 
        """
        self.ϵ, self.δ = ϵ, δ
        self.noise_correction = noise_correction

        self.w, self.d = math.ceil(math.e/ϵ), math.ceil(math.log(1/δ))
        self.X = np.zeros((self.d, self.w), dtype=int)
        print(f"Relative error bound: {self.ϵ} with probability {1-self.δ}.")
        print(f"Instantiated a sketch with size: ({self.d}, {self.w}).")

        # Initialize constants used by hash functions
        self.p = 2**31 - 1
        self.a = np.random.randint(self.p, size=self.d)
        self.b = np.random.randint(self.p, size=self.d)

    def hash(self, x: int):
        """ Computes d pairwise independent hashes of input value x.
            Hash functions implemented as suggested in [2].

            Args:
                x: the input value whose hashes we want to compute
            
            Returns:
                A numpy array of shape (d,) containing d hashes of x.
        """
        return np.mod(np.mod(self.a * x + self.b, self.p), self.w)

    def update(self, x: int, v: int=1):
        assert v >= 0, "Current implementation only supports non-negative updates"
        self.X[range(self.d), self.hash(x)] += v

    def count(self, x: int):
        """
        Compute the count estimate for value x.
        If no noise correction is specified, use the original algorithm in [1,2].
        Else, use one of the variants proposed in [3].
        Note that noise correction does not give necessarily better results.
        """
        
        raw_estimate = min(self.X[range(self.d), self.hash(x)])
        if not self.noise_correction:
            return raw_estimate
        
        if self.noise_correction == 'mean':
            # Estimate noise as average of values of other cells (per each row)
            cardinality = np.sum(self.X[0,:])
            noise = (cardinality - self.X[range(self.d), self.hash(x)])/(self.w - 1)
        
        if self.noise_correction == 'median':
            # Estimate noise as median of values of cells in each row.
            # For simplicity (and speed), not removing the cell of x 
            noise = np.median(self.X, axis=1)

        corrected_estimate = np.median(self.X[range(self.d), self.hash(x)] - noise)
        return np.clip(corrected_estimate, 0, raw_estimate)

    def __getitem__(self, x: int):
        return self.count(x)

    def print_stats(self):
        """ Print some stats. """
        cardinality = np.sum(self.X[0,:])
        error = self.ϵ * cardinality
        probability = 1 - self.δ

        print(f"The total count seen so far is: {cardinality}.")
        print(f"Count estimates have an error ≤ {error} with probability {probability}.")

