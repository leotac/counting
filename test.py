import numpy as np
from matplotlib import pyplot as plt

from collections import Counter
from cmsketch import CountMinSketch
from bloom import BloomFilter

def plot_stats(counts, cm):
    values = np.array(sorted(counts.keys()))
    real_counts = np.array([counts[v] for v in values])
    estimates = np.array([cm[v] for v in values])
    errors = np.abs(estimates - real_counts)
    relative_errors = errors/real_counts
    cm.stats()

    fig, axes = plt.subplots(4)
    axes[0].plot(values, real_counts, label="Real count")
    axes[0].plot(values, estimates, label="Estimated count")
    axes[0].set_xscale('log')
    axes[0].legend()

    axes[1].plot(values, real_counts, label="Real count")
    axes[1].plot(values, estimates, label="Estimated count")
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend()

    axes[2].plot(values, errors, label="Absolute error")
    axes[2].set_xscale('log')
    axes[2].legend()

    axes[3].plot(values, relative_errors, label="Relative error (log scale)")
    axes[3].axhline(1, label="100% error", c='k', linewidth=.5)
    axes[3].axhline(.1, label="10% error", c='k', linewidth=.5)
    axes[3].axhline(.01, label="1% error", c='k', linewidth=.5)
    axes[3].set_xscale('log')
    axes[3].set_yscale('log')
    axes[3].legend()


def test_cminsketch():
    N = 100_000
    a = np.random.zipf(1.3, N)
    counts = Counter(a)
    print(f"The number of distinct values is {len(counts)}")

    cm = CountMinSketch(0.01, 0.01, noise_correction=False)
    for x, c in counts.items():
        cm.update(x, c)
    plot_stats(counts,cm)
    
    cm = CountMinSketch(0.01, 0.01, noise_correction='mean')
    for x, c in counts.items():
        cm.update(x, c)
    plot_stats(counts,cm)

    cm = CountMinSketch(0.01, 0.01, noise_correction='median')
    for x, c in counts.items():
        cm.update(x, c)
    plot_stats(counts,cm)

    plt.show()


def test_bloom():
    # Generate N distinct values in the range [0, 100000]
    N = 50_000
    s = set(np.random.choice(1_000_000, N, replace=False))
    print(f"Num inserted values: {N}, min: {min(s)}, max: {max(s)}")

    desired_error_prob = [0.05, 0.1, 0.2]
    for p in desired_error_prob:
        M, d = BloomFilter.optimal_size(p, N)
        print(f"Desired FP rate: {100*p:.2f}%, size of Bloom filter: "
              f"{M} bits (with {d} hash functions)")
        
        bf = BloomFilter(M, d, s)

        test_set = np.random.choice(1_000_000, 100_000)
        fp, tn = 0, 0
        for x in test_set:
            if x not in s:
                if x in bf:
                    fp += 1
                else:
                    tn += 1
        print(f"Estimated FP rate: {100*fp/(fp + tn):.2f}% (from {fp + tn} negative samples)")
                

if __name__ == '__main__':
    print("##### Count Min Sketch:")
    test_cminsketch()
    
    print("\n##### Bloom filter:")
    test_bloom()

