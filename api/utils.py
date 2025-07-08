import numpy as np

class MixtureDistribution:
    def __init__(self, state_probs, random_variables):
        """
        state_probs: array-like, shape (n,) - probabilities for each state (must sum to 1)
        random_variables: list of callables, each with .rvs(size), .pdf(x), .cdf(x), .logpdf(x), .ppf(q) methods
        """
        self.state_probs = np.array(state_probs)
        # normalize state_probs to ensure they sum to 1
        self.state_probs /= np.sum(self.state_probs)
        self.random_variables = random_variables
        if len(self.state_probs) != len(self.random_variables):
            raise ValueError("state_probs and random_variables must have the same length")
        if not np.isclose(np.sum(self.state_probs), 1):
            raise ValueError("state_probs must sum to 1")

    def rvs(self, size=1):
        """
        Generate random samples from the mixture distribution.
        """
        states = np.random.choice(len(self.state_probs), size=size, p=self.state_probs)
        samples = np.empty(size)
        for i in range(len(self.random_variables)):
            idx = (states == i)
            if np.any(idx):
                samples[idx] = self.random_variables[i].rvs(size=np.sum(idx))
        return samples

    def mean(self):
        """
        Return the mean of the mixture distribution.
        """
        means = np.array([rv.mean() for rv in self.random_variables])
        return np.dot(self.state_probs, means)

    def var(self):
        """
        Return the variance of the mixture distribution.
        """
        means = np.array([rv.mean() for rv in self.random_variables])
        variances = np.array([rv.var() for rv in self.random_variables])
        mean_total = self.mean()
        return np.dot(self.state_probs, variances + (means - mean_total) ** 2)

    def pdf(self, x):
        """
        Compute the PDF at x for the mixture distribution.
        """
        x = np.atleast_1d(x)
        pdf_vals = np.zeros(shape=x.shape[0], dtype=float)
        for p, rv in zip(self.state_probs, self.random_variables):
            # rv.pdf may not be vectorized, so compute for each x
            pdf_vals += p * np.array([rv.pdf(xi) for xi in x])
        return pdf_vals

    def cdf(self, x):
        """
        Compute the CDF at x for the mixture distribution.
        """
        x = np.atleast_1d(x)
        cdf_vals = np.zeros_like(x, dtype=float)
        for p, rv in zip(self.state_probs, self.random_variables):
            cdf_vals += p * rv.cdf(x)
        return cdf_vals

    def logpdf(self, x):
        """
        Compute the log PDF at x for the mixture distribution.
        """
        pdf_val = self.pdf(x)
        return np.log(pdf_val)

    def ppf(self, q, num_points=10000):
        """
        Approximate the quantile function (inverse CDF) using sampling.
        """
        samples = self.rvs(size=num_points)
        samples.sort()
        idx = (q * (num_points - 1)).astype(int)
        return samples[idx]
    
    def max_idx(self, min_val=-10, max_val=10, num_points=1000):
        """
        Return the x value where the PDF is maximized.
        """
        x = np.linspace(min_val, max_val, num_points)
        pdf_vals = self.pdf(x)
        return x[np.argmax(pdf_vals)]

def test_mixture_distribution():
    from scipy.stats import norm, uniform

    state_probs = [0.7, 0.3]
    random_variables = [uniform(loc=-3, scale=2), norm(loc=1, scale=0.8)]

    md = MixtureDistribution(state_probs, random_variables)
    print("Mean:", md.mean())
    print("Variance:", md.var())
    print("Sample rvs:", md.rvs(size=5))
    print("PDF at 0:", md.pdf(0))
    print("CDF at 0.5:", md.cdf(0.5))

    import matplotlib.pyplot as plt

    x = np.linspace(-4, 4, 500)
    pdf_vals = md.pdf(x)
    cdf_vals = md.cdf(x)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x, pdf_vals, label="Mixture PDF")
    max_x = md.max_idx(min_val=x[0], max_val=x[-1], num_points=len(x))
    plt.axvline(max_x, color="red", linestyle="--", label=f"max_idx={max_x:.2f}")
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.title("Mixture Distribution PDF")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, cdf_vals, label="Mixture CDF", color="orange")
    plt.xlabel("x")
    plt.ylabel("CDF")
    plt.title("Mixture Distribution CDF")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_mixture_distribution()
