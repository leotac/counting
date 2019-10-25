import numpy as np

class CountMinSketch:

    def __init__(self, d: int, w: int):
        """
        Keeps an approximate count of each distinct values.
    
        [1] G. Cormode, S. Muthukrishnan. An Improved Data Stream Summary:
                                          The Count-Min Sketch and its Applications
        [2] G. Cormode, S. Muthukrishnan. Approximating Data with the Count-Min Data Structure
        """
        self.d, self.w = d, w
        self.X = np.zeros((d, w), dtype=int)
        self.p = 2**31 - 1
        self.a = np.random.randint(self.p, size=d)
        self.b = np.random.randint(self.p, size=d)

    def hash(self, x: int):
        """ Returns all d hashes of value x
            Hash functions as suggested in [2]
        """
        return np.mod(np.mod(self.a * x + self.b, self.p), self.w)

    def update(self, x: int, v: int=1):
        self.X[range(self.d), self.hash(x)] += v

    def count(self, x: int):
        return min(self.X[range(self.d), self.hash(x)])

    def __getitem__(self, x: int):
        return self.count(x)



if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from collections import Counter
    a = np.random.zipf(1.3, 100_000)
    counts = Counter(a)

    cm = CountMinSketch(3, 100)
    for x in a:
        cm.update(x)

    values = sorted(counts)
    real_counts = [counts[v] for v in values]
    estimates = [cm[v] for v in values]

    plt.plot(values[:100], real_counts[:100])
    plt.plot(values[:100], estimates[:100])
    plt.show()

