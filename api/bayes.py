import numpy as np
from scipy.stats import invgamma, multivariate_normal, t
from scipy.stats import invwishart, matrix_normal, multivariate_t
class BayesianMultivariateNormalEstimator:
    def __init__(self, mu, kappa, nu, psi):
        """
        Initialize the estimator with NIW prior parameters.

        Parameters:
        - mu: Prior mean (d-dimensional array)
        - kappa: Prior scaling factor (scalar)
        - nu: Prior degrees of freedom (scalar, > d - 1)
        - psi: Prior scale matrix (d x d array)
        """
        self.mu = mu
        self.kappa = kappa
        self.nu = nu
        self.psi = psi
        self.d = mu.shape[0]

    def fit(self, X):
        """
        Perform Bayesian estimation using Normal-Inverse-Wishart prior.
        Handles missing values (np.nan) by using only observed entries for each sample.

        Parameters:
        - X: np.array of shape (n_samples, n_features)
        """
        X = np.asarray(X)
        n, d = X.shape
        assert self.mu.shape[0] == d, "Dimension mismatch between prior mean and data."

        mask = ~np.isnan(X)
        n_obs = np.sum(mask, axis=0)
        x_bar = np.nanmean(X, axis=0)

        S = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                xi = X[:, i]
                xj = X[:, j]
                valid = ~np.isnan(xi) & ~np.isnan(xj)
                if np.any(valid):
                    S[i, j] = np.dot(xi[valid] - x_bar[i], xj[valid] - x_bar[j])
                else:
                    S[i, j] = 0.0

        n_eff = np.sum(np.all(mask, axis=1))
        n_used = n_eff if n_eff > 0 else np.max(n_obs)

        kappa_new = self.kappa + n_used
        nu_new = self.nu + n_used
        mu_new = (self.kappa * self.mu + n_used * x_bar) / kappa_new
        delta = (x_bar - self.mu).reshape(-1, 1)
        psi_new = self.psi + S + (self.kappa * n_used / kappa_new) * np.dot(delta, delta.T)

        self.mu = mu_new
        self.kappa = kappa_new
        self.nu = nu_new
        self.psi = psi_new

    def summary(self):
        """
        Print the posterior estimated parameters.
        """
        print("#" + "-" * 20 + "#")
        print("Posterior Parameters (Normal-Inverse-Wishart):")
        print("#" + "-" * 20 + "#")
        print(f"  mu (mean): {self.mu}")
        print(f"  kappa (mean precision): {self.kappa}")
        print(f"  nu (degrees of freedom): {self.nu}")
        print(f"  psi (scale matrix):\n{self.psi}")
        print("#" + "-" * 20 + "#")

    def sample(self, prior=True):
        """
        Sample from the prior or posterior predictive distribution.

        Parameters:
        - prior: If True, sample from the prior; otherwise, sample from the posterior.

        Returns:
        - mu_sample: Sampled mean vector (d-dimensional array)
        - Sigma: Sampled covariance matrix (d x d array)
        """
        if prior:
            mu = self.mu
            kappa = self.kappa
            nu = self.nu
            psi = self.psi
        else:
            mu = self.mu
            kappa = self.kappa
            nu = self.nu
            psi = self.psi

        Sigma = invwishart.rvs(df=nu, scale=psi)
        mu_sample = multivariate_normal.rvs(mean=mu, cov=Sigma / kappa)
        return mu_sample, Sigma

    def estimate(self):
        if self.mu is None or self.kappa is None or self.nu is None or self.psi is None:
            raise ValueError("Model must be fit before estimating.")
        mu = self.mu
        scale = (self.kappa + 1) / (self.kappa * (self.nu - self.d + 1)) * self.psi
        return mu, scale

    def predict(self, X_new):
        """
        Predict mean and covariance for new data points.
        X_new has missing values, and the goal is to predict the mean and covariance of the posterior distribution observing the data.

        Parameters:
        - X_new: np.array of shape (n_samples, d)

        Returns:
        - mean: Predictive mean (n_samples, d)
        - cov: Predictive covariance matrix (d x d) for each sample (list of arrays)
        """
        if self.mu is None or self.psi is None or self.kappa is None or self.nu is None:
            raise ValueError("Model must be fit before prediction.")

        mu = self.mu
        Sigma = self.psi / (self.nu - self.d - 1)

        X_new = np.atleast_2d(X_new)
        n_samples, d = X_new.shape
        assert d == self.d, "Input dimension mismatch."

        means = []
        covs = []
        for i in range(n_samples):
            x = X_new[i]
            obs_idx = np.where(~np.isnan(x))[0]
            miss_idx = np.where(np.isnan(x))[0]

            if len(miss_idx) == 0:
                means.append(mu)
                covs.append(Sigma)
                continue

            mu_o = mu[obs_idx]
            mu_m = mu[miss_idx]

            Sigma_oo = Sigma[np.ix_(obs_idx, obs_idx)]
            Sigma_om = Sigma[np.ix_(obs_idx, miss_idx)]
            Sigma_mo = Sigma[np.ix_(miss_idx, obs_idx)]
            Sigma_mm = Sigma[np.ix_(miss_idx, miss_idx)]

            x_o = x[obs_idx]

            Sigma_oo_inv = np.linalg.inv(Sigma_oo)
            cond_mean = mu_m + Sigma_mo @ Sigma_oo_inv @ (x_o - mu_o)
            cond_cov = Sigma_mm - Sigma_mo @ Sigma_oo_inv @ Sigma_om

            means.append(cond_mean)
            covs.append(cond_cov)

        means = np.stack(means, axis=0)
        return means, covs

    def posterior(self, x):
        """
        Computes the posterior predictive density of new observation x, handling missing values (np.nan).

        Parameters:
        - x: (d,) numpy array (a new data point, possibly with np.nan)

        Returns:
        - pdf: posterior predictive probability density at x (over observed dimensions)
        """
        if self.mu is None:
            raise ValueError("Model must be fit before evaluating predictive probability.")

        x = np.asarray(x)
        if x.ndim != 1 or x.shape[0] != self.d:
            raise ValueError(f"x must be a 1D array of shape ({self.d},)")

        obs_idx = np.where(~np.isnan(x))[0]
        if len(obs_idx) == 0:
            raise ValueError("At least one observed value is required in x.")

        x_obs = x[obs_idx]
        mu_obs = self.mu[obs_idx]
        psi_obs = self.psi[np.ix_(obs_idx, obs_idx)]

        kappa = self.kappa
        nu = self.nu
        d_obs = len(obs_idx)
        df = nu - self.d + 1

        scale = (kappa + 1) / (kappa * df) * psi_obs

        pdf = multivariate_t.pdf(x_obs, df=df, loc=mu_obs, shape=scale)
        return pdf

class BayesianLinearRegression:
    def __init__(self, beta, Sigma, a, b):
        """
        Prior parameters:
        - beta: Prior mean for beta (shape: d,)
        - Sigma: Prior covariance matrix for beta (shape: d x d)
        - a, b: Hyperparameters for Inverse-Gamma prior on sigma^2
        """
        self.beta = beta
        self.Sigma = Sigma
        self.a = a
        self.b = b

    def fit(self, X, y):
        """
        Fit the Bayesian linear regression model to data.
        Handles missing values (np.nan) in X or y by using only fully observed rows.

        Inputs:
        - X: (n x d) input matrix
        - y: (n,) output vector
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n, d = X.shape
        assert self.beta.shape[0] == d
        assert self.Sigma.shape == (d, d)

        # Mask for fully observed rows (no nan in X or y)
        mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        X_obs = X[mask]
        y_obs = y[mask]
        n_obs = X_obs.shape[0]

        if n_obs == 0:
            raise ValueError("No fully observed data rows available for fitting.")

        Sigma_inv = np.linalg.inv(self.Sigma)
        XtX = X_obs.T @ X_obs
        Xty = X_obs.T @ y_obs

        Sigma_post = np.linalg.inv(Sigma_inv + XtX)
        beta_post = Sigma_post @ (Sigma_inv @ self.beta + Xty)

        a_post = self.a + n_obs / 2
        b_post = self.b + 0.5 * (
            y_obs @ y_obs +
            self.beta.T @ Sigma_inv @ self.beta -
            beta_post.T @ np.linalg.inv(Sigma_post) @ beta_post
        )

        self.beta = beta_post
        self.Sigma = Sigma_post
        self.a = a_post
        self.b = b_post

    def summary(self):
        """
        Print posterior parameters.
        """
        if self.beta is None:
            print("Model is not yet fit.")
            return

        print("#" + "-" * 20 + "#")
        print("Posterior for beta ~ N(beta, sigma^2 * Sigma):")
        print("#" + "-" * 20 + "#")
        print("beta =", self.beta)
        print("Sigma =\n", self.Sigma)
        print(f"Posterior for sigma^2 ~ Inverse-Gamma(a={self.a}, b={self.b})")
        print("#" + "-" * 20 + "#")

    def sample(self):
        """
        Draw samples from the joint posterior of (beta, sigma^2).
        
        Returns:
        - beta: Sampled regression coefficients (shape: d,)
        - sigma2: Sampled variance (scalar)
        """
        sigma2 = invgamma.rvs(self.a, scale=self.b)
        beta = multivariate_normal.rvs(mean=self.beta, cov=sigma2 * self.Sigma)
        return beta, sigma2

    def predict(self, x_star, return_std=False):
        """
        Predictive distribution at new point x_star (shape: d,).
        Handles missing values (np.nan) in x_star by conditioning on observed features.
        Returns mean and (optionally) std of Student-t predictive distribution.
        """
        if self.beta is None:
            raise ValueError("Model must be fit before prediction.")

        x_star = np.asarray(x_star).flatten()
        obs_idx = np.where(~np.isnan(x_star))[0]
        if len(obs_idx) == 0:
            raise ValueError("At least one observed value is required in x_star.")

        x_obs = x_star[obs_idx].reshape(1, -1)
        beta_obs = self.beta[obs_idx]
        Sigma_obs = self.Sigma[np.ix_(obs_idx, obs_idx)]

        mean = x_obs @ beta_obs
        scale = (self.b / self.a) * (1 + x_obs @ Sigma_obs @ x_obs.T)
        std = np.sqrt(scale)
        df = 2 * self.a

        if return_std:
            return mean.item(), std.item(), df
        else:
            return t(df=df, loc=mean.item(), scale=std.item())

    def estimate(self):
        # return mean of multivariate_normal and mean of Inverse-Gamma
        if self.beta is None:
            raise ValueError("Model must be fit before estimating.")
        if self.Sigma is None:
            raise ValueError("Model must be fit before estimating.")
        if self.a is None:
            raise ValueError("Model must be fit before estimating.")
        if self.b is None:
            raise ValueError("Model must be fit before estimating.")
        # mean of multivariate_normal
        mu = self.beta
        # mean of Inverse-Gamma
        scale = self.b / (self.a - 1)
        return mu, scale

    def posterior(self, x, y):
        """
        Compute the posterior predictive probability of a single observation (x, y).
        Handles missing values (np.nan) in x by conditioning on observed features.

        Parameters:
        - x: (d,) feature vector (may contain np.nan)
        - y: scalar target

        Returns:
        - pdf: scalar, posterior predictive density of y given x
        """
        if self.beta is None:
            raise ValueError("Model must be fit before calling posterior")

        x = np.asarray(x).flatten()
        obs_idx = np.where(~np.isnan(x))[0]
        if len(obs_idx) == 0:
            raise ValueError("At least one observed value is required in x.")

        x_obs = x[obs_idx].reshape(1, -1)
        beta_obs = self.beta[obs_idx]
        Sigma_obs = self.Sigma[np.ix_(obs_idx, obs_idx)]

        mean = float(x_obs @ beta_obs)
        scale = (self.b / self.a) * (1 + x_obs @ Sigma_obs @ x_obs.T)
        scale = float(scale)
        df = 2 * self.a

        return t.pdf(y, df=df, loc=mean, scale=np.sqrt(scale))

class BayesianMultivariateLinearRegression:
    def __init__(self, M, V, S, nu):
        """
        Prior parameters:
        - M: Prior mean for beta (shape: m x k)
        - V: Prior row covariance (shape: m x m)
        - S: Prior scale matrix for Sigma (shape: k x k)
        - nu: Degrees of freedom for Inverse-Wishart prior on Sigma
        """
        self.M = M
        self.V = V
        self.S = S
        self.nu = nu

    def fit(self, X, Y):
        """
        Fit the Bayesian multivariate linear regression model.
        Handles missing values (np.nan) by using only fully observed rows.

        Inputs:
        - X: (n x m) input matrix
        - Y: (n x k) output matrix
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        n, m = X.shape
        _, k = Y.shape

        # Only use rows where all X and Y are observed
        mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(Y).any(axis=1))
        X_obs = X[mask]
        Y_obs = Y[mask]
        n_obs = X_obs.shape[0]

        if n_obs == 0:
            raise ValueError("No fully observed data rows available for fitting.")

        V_inv = np.linalg.inv(self.V)
        XtX = X_obs.T @ X_obs
        XtY = X_obs.T @ Y_obs

        V_new_inv = V_inv + XtX
        self.V = np.linalg.inv(V_new_inv)
        self.M = self.V @ (V_inv @ self.M + XtY)

        self.nu = self.nu + n_obs

        YtY = Y_obs.T @ Y_obs
        MtVinvM = self.M.T @ V_inv @ self.M
        MnewtVnewinvMnew = self.M.T @ V_new_inv @ self.M

        self.S = self.S + YtY + MtVinvM - MnewtVnewinvMnew

    def sample(self):
        """
        Sample from the posterior distribution of (beta, Sigma).
        Returns:
        - beta: Sampled coefficient matrix (m x k)
        - Sigma: Sampled covariance matrix (k x k)
        """
        Sigma = invwishart.rvs(df=self.nu, scale=self.S)
        beta = matrix_normal.rvs(mean=self.M, rowcov=self.V, colcov=Sigma)
        return beta, Sigma

    def estimate(self):
        """
        Return posterior mean estimates of beta and Sigma.
        """
        if self.M is None or self.V is None or self.S is None or self.nu is None:
            raise ValueError("Model must be fit before estimating.")
        
        beta_mean = self.M
        Sigma_mean = self.S / (self.nu - self.M.shape[1] - 1)
        return beta_mean, Sigma_mean

    def posterior(self, x, y):
        """
        Compute the posterior predictive density for a new observation (x, y).
        Handles missing values (np.nan) in x or y by conditioning on observed features/outputs.
        Inputs:
        - x: (m,) new input vector (may contain np.nan)
        - y: (k,) new output vector (may contain np.nan)
        Returns:
        - pdf: Posterior predictive density over observed outputs/features
        """
        if self.M is None or self.V is None or self.S is None or self.nu is None:
            raise ValueError("Model must be fit before calling posterior.")

        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        m, k = self.M.shape

        # Find observed indices in x and y
        obs_x = np.where(~np.isnan(x))[0]
        obs_y = np.where(~np.isnan(y))[0]

        if len(obs_x) == 0 or len(obs_y) == 0:
            raise ValueError("At least one observed value is required in both x and y.")

        # Subset x, M, V, S, etc. to observed indices
        x_obs = x[obs_x].reshape(1, -1)
        M_obs = self.M[np.ix_(obs_x, obs_y)]  # shape: (len(obs_x), len(obs_y))
        V_obs = self.V[np.ix_(obs_x, obs_x)]
        S_obs = self.S[np.ix_(obs_y, obs_y)]
        k_obs = len(obs_y)
        nu = self.nu

        # Posterior predictive mean and covariance for observed outputs
        Sigma_mean = S_obs / (nu - k_obs - 1)
        mean = x_obs @ M_obs  # shape: (1, k_obs)
        cov = Sigma_mean + (x_obs @ V_obs @ x_obs.T)[0, 0] * Sigma_mean

        y_obs = y[obs_y]

        return multivariate_normal.pdf(y_obs, mean=mean.flatten(), cov=cov)

    def predict(self, X_new, return_std=False):
        """
        Predict the mean response for new inputs X_new, handling missing values (np.nan).
        Inputs:
        - X_new: (n_new x m) matrix of new inputs (may contain np.nan)
        Returns:
        - mean: (n_new x k) matrix of predicted means (np.nan where not computable)
        - cov: list of (k x k) predictive covariance matrices for each sample
        """
        if self.M is None or self.V is None or self.S is None or self.nu is None:
            raise ValueError("Model must be fit before prediction.")

        X_new = np.atleast_2d(X_new)
        n_new, m = X_new.shape
        k = self.M.shape[1]
        means = []
        covs = []
        for i in range(n_new):
            x = X_new[i]
            obs_idx = np.where(~np.isnan(x))[0]
            if len(obs_idx) == 0:
                means.append(np.full(k, np.nan))
                covs.append(np.full((k, k), np.nan))
                continue
            x_obs = x[obs_idx].reshape(1, -1)
            M_obs = self.M[obs_idx, :]
            V_obs = self.V[np.ix_(obs_idx, obs_idx)]
            Sigma_mean = self.S / (self.nu - self.M.shape[1] - 1)
            mean = x_obs @ M_obs
            cov = Sigma_mean + (x_obs @ V_obs @ x_obs.T)[0, 0] * Sigma_mean
            means.append(mean.flatten())
            covs.append(cov)
        means = np.stack(means, axis=0)
        if return_std:
            return means, covs
        else:
            # if there is only one sample, return a single covariance matrix
            if len(covs) > 1:
                raise ValueError("Multiple samples provided, cannot return single covariance matrix.")
            return multivariate_t(df=self.nu - self.M.shape[1] - 1, loc=means.flatten(), shape=covs[0])

    def summary(self):
        """
        Print posterior parameters summary.
        """
        if self.M is None:
            print("Model is not yet fit.")
            return

        print("#" + "-" * 20 + "#")
        print("Posterior for beta ~ MatrixNormal(M, V, Sigma):")
        print("#" + "-" * 20 + "#")
        print("M =\n", self.M)
        print("V =\n", self.V)
        print("Posterior for Sigma ~ Inverse-Wishart(nu={}, S):".format(self.nu))
        print("S =\n", self.S)
        print("#" + "-" * 20 + "#")

def test_BayesianMultivariateNormalEstimator():
    # Simulate data
    np.random.seed(0)
    X = np.random.multivariate_normal(
        mean=[1, 3, 2],
        cov=np.array([[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 1.5]]),
        size=100
    )

    # Set priors
    mu_0 = np.zeros(3)
    kappa_0 = 1.0
    nu_0 = 5.0
    psi_0 = np.eye(3)

    # Fit estimator
    model = BayesianMultivariateNormalEstimator(mu_0, kappa_0, nu_0, psi_0)
    model.fit(X)
    model.summary()

    # Sample from prior
    prior_samples = model.sample(prior=True)
    print("Samples from prior:")
    print(prior_samples)
    
    # test estimate
    mu, scale = model.estimate()
    print("Estimated mean:", mu)
    print("Estimated scale matrix:", scale)

    # Sample from posterior
    posterior_samples = model.sample(prior=False)
    print("Samples from posterior:")
    print(posterior_samples)

    # Predict at a new point
    x_star = np.array([1.15, np.nan, np.nan])
    mean, cov = model.predict(x_star)
    pdf = model.posterior(x_star)
    print("Posterior predictive density at x_star:", pdf)
    print("Predictive mean at x_star:", mean)
    print("Predictive covariance at x_star:\n", cov)

def test_BayesianLinearRegression():
    # Generate synthetic data
    np.random.seed(0)
    X = np.random.randn(100, 2)
    true_beta = np.array([10.0, -3.0])
    y = X @ true_beta + np.random.randn(100) * 0.5

    # Define priors
    beta_0 = np.zeros(2)
    Sigma_0 = np.eye(2)
    a_0 = 2.0
    b_0 = 2.0

    # Fit model
    blr = BayesianLinearRegression(beta_0, Sigma_0, a_0, b_0)
    blr.fit(X, y)
    blr.summary()

    #test estimate
    beta, scale = blr.estimate()
    print("Estimated beta:", beta)
    print("Estimated scale:", scale)

    # Predict at new point
    x_star = np.array([3.0, 2.0])
    y_star = x_star @ true_beta
    mean, std, df = blr.predict(x_star, return_std=True)
    print(f"Predictive mean = {mean:.3f}, std = {std:.3f}, df = {df}")
    print(f"Actual mean = {y_star:.3f}")

def test_BayesianMultivariateLinearRegression():
    # Set random seed for reproducibility
    np.random.seed(0)

    # Generate synthetic data
    n, m, k = 200, 2, 3  # 100 samples, 2 features, 3 outputs
    X = np.random.randn(n, m)
    true_beta = np.array([[2.0, -1.0, 0.5], [1.0, 0.5, -1.5]])  # shape: (2 x 3)
    true_Sigma = np.array([[1.0, 0.3, 0.2],
                           [0.3, 1.5, 0.4],
                           [0.2, 0.4, 2.0]])  # shape: (3 x 3)

    E = np.random.multivariate_normal(np.zeros(k), true_Sigma, size=n)
    Y = X @ true_beta + E  # shape: (100 x 3)

    # Define priors
    M0 = np.zeros((m, k))
    V0 = np.eye(m)
    S0 = np.eye(k)
    nu0 = k + 2  # Must be > k - 1 for prior to be proper

    # Fit model
    model = BayesianMultivariateLinearRegression(M0, V0, S0, nu0)
    model.fit(X, Y)
    model.summary()

    # Estimate posterior means
    beta_est, Sigma_est = model.estimate()
    print("Estimated beta:\n", beta_est)
    print("Estimated Sigma:\n", Sigma_est)

    # Predict at a new point
    x_star = np.array([[3.0, -4.0]])  # shape: (1 x 2)
    y_true = x_star @ true_beta  # true value for comparison
    mean_pred, cov_pred = model.predict(x_star, return_std=True)
    print("Predictive mean:\n", mean_pred)
    print("Predictive covariance:\n", cov_pred)
    print("True mean:\n", y_true)

    # Posterior predictive density
    y_star = y_true.flatten()
    pdf = model.posterior(x_star.flatten(), y_star)
    print(f"Posterior predictive density: {pdf:.3f}")


if __name__ == "__main__":
    #test_BayesianMultivariateNormalEstimator()
    test_BayesianLinearRegression()
    #test_BayesianMultivariateLinearRegression()