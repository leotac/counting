import numpy as np

class Hash:
    
    def __init__(self, d: int, m: int, p: int=2**31 - 1):
        """
        Create callable that computes d pairwise independent hashes of an input value x.
        Hash functions from ℕ → [0,m] implemented as suggested in [1].
        
        [1] G. Cormode, S. Muthukrishnan. Approximating Data with the Count-Min Data Structure
        """
        self.d = d
        self.m = m
        self.p = p 
        self.a = np.random.randint(self.p, size=self.d)
        self.b = np.random.randint(self.p, size=self.d)
    
    def __call__(self, x: int):
        """ Computes d pairwise independent hashes of input value x.

            Args:
                x: the input value whose hashes we want to compute
            
            Returns:
                A numpy array of shape (d,) containing d independent hashes of x.
        """
        return np.mod(np.mod(self.a * x + self.b, self.p), self.m)

