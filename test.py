import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

def plot_stats(counts, cm):
    values = np.array(sorted(counts.keys()))
    real_counts = np.array([counts[v] for v in values])
    estimates = np.array([cm[v] for v in values])
    errors = np.abs(estimates - real_counts)
    relative_errors = errors/real_counts
    cm.print_stats()
    
   
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


if __name__ == '__main__':
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
