import numpy as np
from tqdm import tqdm
from scipy.stats import beta
from typing import Literal
from api.bayes import BayesianMultivariateNormalEstimator
from api.bayes import BayesianLinearRegression
from api.bayes import BayesianMultivariateLinearRegression
from scipy.stats import multivariate_normal
from api.utils import MixtureDistribution

class StickyHDPHMM:
    def __init__(self, ups=1.0, gamma=1.0, rho=0.5, weight = 1.0, kappa_0 = 1.0, nu_0 = 5.0, a_0 = 2.0, b_0 = 2.0, 
                 model: Literal['multivariate', 'linear', 'multilinear'] = 'multivariate'):
        self.rho = rho
        self.ups = ups
        self.gamma = gamma
        self.kappa = rho * ups
        self.alpha = ups * (1 - rho)
        self.model = model
        self.weight = weight
        self.kappa_0 = kappa_0
        self.nu_0 = nu_0
        self.a_0 = a_0
        self.b_0 = b_0

    @property
    def params(self):
        """Return the parameters of the model."""
        return {
            'rho': self.rho,
            'ups': self.ups,
            'gamma': self.gamma,
            'model': self.model,
            'weight': self.weight,
            'kappa_0': self.kappa_0,
            'nu_0': self.nu_0,
            'a_0': self.a_0,
            'b_0': self.b_0
        }

    def _init_estimator(self, x, y=None):
        """Initialize the cluster/state estimator based on the model type."""
        if self.model == 'multivariate':
            d = x.shape[0] if x.ndim == 1 else x.shape[1]
            mu_0 = np.zeros(d)
            kappa_0 = self.kappa_0
            nu_0 = self.nu_0
            psi_0 = np.eye(d)
            return BayesianMultivariateNormalEstimator(mu_0, kappa_0, nu_0, psi_0)
        elif self.model == 'linear':
            d = x.shape[0] if x.ndim == 1 else x.shape[1]
            beta_0 = np.zeros(d)
            Sigma_0 = np.eye(d)
            a_0 = self.a_0
            b_0 = self.b_0

            mu_0 = np.zeros(d)
            kappa_0 = self.kappa_0
            nu_0 = self.nu_0
            psi_0 = np.eye(d)
            # Assuming BayesianLinearRegression is imported
            return BayesianLinearRegression(beta_0, Sigma_0, a_0, b_0), BayesianMultivariateNormalEstimator(mu_0, kappa_0, nu_0, psi_0)
        elif self.model == 'multivariate_linear':
            m = x.shape[0] if x.ndim == 1 else x.shape[1]
            k = y.shape[0] if y is not None else 1
            M0 = np.zeros((m, k))
            V0 = np.eye(m)
            S0 = np.eye(k)
            nu0 = k + 2

            d = x.shape[0] if x.ndim == 1 else x.shape[1]
            mu_0 = np.zeros(d)
            kappa_0 = self.kappa_0
            nu_0 = self.nu_0
            psi_0 = np.eye(d)
            # Assuming BayesianMultivariateLinearRegression is imported
            return BayesianMultivariateLinearRegression(M0, V0, S0, nu0), BayesianMultivariateNormalEstimator(mu_0, kappa_0, nu_0, psi_0)
        else:
            raise ValueError(f"Unknown model type: {self.model}")

    def _initialize(self, data):
        """
        Initialize state assignments, transition counts, beta weights, and estimators.
        """
        self.T = len(data)
        if self.model == 'multivariate':
            self.d = data.shape[1] if data.ndim > 1 else 1
        elif self.model == 'linear':
            self.d = data.shape[1] - 1 if data.ndim > 1 else 1
        elif self.model == 'multivariate_linear':
            self.d = data[0][0].shape[0]
        else:
            raise ValueError(f"Unknown model type: {self.model}")

        # Initialization using CRP (Chinese Restaurant Process)
        z = []
        table_counts = []
        for t in range(self.T):
            if t == 0 or np.random.rand() < self.alpha / (self.alpha + t):
                # New table
                z.append(len(table_counts))
                table_counts.append(1)
            else:
                probs = np.array(table_counts) / (self.alpha + t)
                k = np.random.choice(len(table_counts), p=probs / probs.sum())
                z.append(k)
                table_counts[k] += 1
        self.z = np.array(z) if self.init_z is None else np.array(self.init_z).flatten()
        self.K = len(table_counts)
        betas = np.random.dirichlet(table_counts + [self.gamma])
        self.beta_tilde = betas[-1]
        self.betas = betas[:-1]
        self.estimators = []
        for _ in range(self.K):
            if self.model == 'multivariate':
                self.estimators.append(self._init_estimator(np.zeros(self.d)))
            elif self.model == 'linear':
                self.estimators.append(self._init_estimator(np.zeros(self.d)))
            elif self.model == 'multivariate_linear':
                x_shape = data[0][0].shape
                y_shape = data[0][1].shape
                self.estimators.append(self._init_estimator(np.zeros(x_shape), np.zeros(y_shape)))
        self.n = np.zeros((self.K, self.K), dtype=int)
        # Initialize transition counts
        for t in range(1, self.T):
            self.n[self.z[t-1], self.z[t]] += 1

    def _calculate_posterior(self, k, x, y=None):
        """Calculate the posterior probabilities for each cluster."""
        if self.model == 'multivariate':
            est = self.estimators[k]
            prob = est.posterior(x)
        else:
            est = self.estimators[k]
            proby = est[0].posterior(x, y)
            probx = est[1].posterior(x)
            prob = proby ** self.weight * probx ** (1-self.weight)
        return prob

    def _update_estimator(self, estimator, data, state):
        """Update the estimator with the current cluster data."""
        if self.model == 'multivariate':
            # data: array of shape (T, d)
            cluster_data = data[self.z == state]
            # initialize a new estimator
            estimator = self._init_estimator(cluster_data[0]) if cluster_data.shape[0] > 0 else estimator
            estimator.fit(cluster_data)
            return estimator
        elif self.model == 'linear':
            # data: array of shape (T, d+1), last column is y
            cluster_data = data[self.z == state]
            if cluster_data.shape[0] == 0:
                return estimator
            x = cluster_data[:, :-1]
            y = cluster_data[:, -1]
            estimator = self._init_estimator(x[0], y[0]) if x.shape[0] > 0 else estimator
            estimator[0].fit(x, y)
            estimator[1].fit(x)
            return estimator
        elif self.model == 'multivariate_linear':
            # data: list of (x, y) tuples
            cluster_data = [pair for idx, pair in enumerate(data) if self.z[idx] == state]
            if len(cluster_data) == 0:
                return estimator
            x = np.stack([pair[0] for pair in cluster_data])
            y = np.stack([pair[1] for pair in cluster_data])
            estimator = self._init_estimator(x[0], y[0]) if len(cluster_data) > 0 else estimator
            estimator[0].fit(x, y)
            estimator[1].fit(x)
            return estimator
        else:
            raise ValueError(f"Unknown model type: {self.model}")


    def fit_one_step(self, data):
        # Rao-Blackwellized Gibbs Sampler for Sticky HDP-HMM
        # 1. For each time step t = 0, ..., T-1
        for t in range(self.T):
            z_prev = self.z[t-1] if t > 0 else None
            z_next = self.z[t+1] if t < self.T-1 else None
            z_old = self.z[t]

            # (a) Remove previous assignment
            if t > 0:
                self.n[z_prev, z_old] -= 1
            if t < self.T-1:
                self.n[z_old, z_next] -= 1
            
            # (b) Compute assignment probabilities for all current states and a new state
            K = self.K
            probs = []
            for k in range(K+1):
                # Transition prior
                if k < K:
                    # Existing state
                    n_prevk = self.n[z_prev, k] if (z_prev is not None) else 0
                    # n_k_next is not used
                    n_k_dot = self.n[k].sum() if k < self.n.shape[0] else 0

                    delta_prevk = int(z_prev == k) if z_prev is not None else 0
                    delta_knext = int(k == z_next) if z_next is not None else 0
                    delta_prevk_knext = delta_prevk * delta_knext

                    numer = self.alpha * self.betas[k] + n_prevk + self.kappa * delta_prevk
                    denom = self.alpha + n_k_dot + self.kappa + delta_prevk

                    numer2 = self.alpha * self.betas[z_next] if (z_next is not None and z_next < K) else 0
                    numer2 += self.n[k, z_next] if (z_next is not None and k < self.n.shape[0] and z_next < self.n.shape[1]) else 0
                    numer2 += self.kappa * delta_knext
                    numer2 += delta_prevk_knext

                    trans_prob = numer * (numer2 / denom) if denom > 0 else 0
                else:
                    # New state
                    numer = self.alpha ** 2 * self.beta_tilde
                    numer2 = self.betas[z_next] if (z_next is not None and z_next < K) else 0
                    trans_prob = numer * numer2 / (self.alpha + self.kappa) if (self.alpha + self.kappa) > 0 else 0

                # Likelihood
                if k < K:
                    # use _calculate_posterior to obtain the likelihood, fallback to 0 if error
                    try:
                        if self.model == 'multivariate':
                            ll = self._calculate_posterior(k, data[t])
                        elif self.model == 'linear':
                            x_t = data[t, :-1]
                            y_t = data[t, -1]
                            ll = self._calculate_posterior(k, x_t, y_t)
                        elif self.model == 'multivariate_linear':
                            x_t, y_t = data[t]
                            ll = self._calculate_posterior(k, x_t, y_t)
                        else:
                            ll = 0
                    except Exception:
                        ll = 0
                else:
                    # For new state, use prior predictive
                    if self.model == 'multivariate':
                        prior_est = self._init_estimator(np.zeros(self.d))
                        ll = prior_est.posterior(data[t])
                    elif self.model == 'linear':
                        prior_est = self._init_estimator(np.zeros(self.d))
                        x_t = data[t, :-1]
                        y_t = data[t, -1]
                        ll = prior_est[0].posterior(x_t, y_t) * prior_est[1].posterior(x_t)
                    elif self.model == 'multivariate_linear':
                        x_t, y_t = data[t]
                        prior_est = self._init_estimator(np.zeros_like(x_t), np.zeros_like(y_t))
                        ll = prior_est[0].posterior(x_t, y_t) * prior_est[1].posterior(x_t)
                    else:
                        ll = 0

                probs.append(trans_prob * ll)

            # Normalize and sample new state
            probs = np.array(probs)
            
            # handle case where we have nan in probs
            if np.isnan(probs).any():
                probs = np.nan_to_num(probs, nan=0.0)

            # handle case where all probabilities are zero
            if probs.sum() == 0:
                probs = np.ones(K + 1)
                probs[K] = 1e-10  # small probability for new state

            probs = probs / probs.sum()
            k_new = np.random.choice(K+1, p=probs)

            # (d) If new state, increment K and split beta
            if k_new == K:
                # Add new state
                self.K += 1
                b = beta.rvs(1, self.gamma)
                beta_new = b * self.beta_tilde
                self.betas = np.append(self.betas, beta_new)
                self.beta_tilde = (1 - b) * self.beta_tilde
                # Add new estimator
                if self.model == 'multivariate':
                    self.estimators.append(self._init_estimator(np.zeros(self.d)))
                elif self.model == 'linear':
                    self.estimators.append(self._init_estimator(np.zeros(self.d)))
                elif self.model == 'multivariate_linear':
                    x_shape = data[0][0].shape
                    y_shape = data[0][1].shape
                    self.estimators.append(self._init_estimator(np.zeros(x_shape), np.zeros(y_shape)))
                # Expand transition matrix
                n_new = np.zeros((self.K, self.K), dtype=int)
                n_new[:self.n.shape[0], :self.n.shape[1]] = self.n
                self.n = n_new

            # Assign new state
            self.z[t] = k_new

            # if t is the last time step, update z to be z_prev
            if t == self.T - 1:
                self.z[t] = self.z[t - 1]

            # (e) Update transition counts
            if t > 0:
                self.n[self.z[t-1], self.z[t]] += 1
            if t < self.T-1:
                self.n[self.z[t], self.z[t+1]] += 1

            # 3.1 Remove empty states according to: remove j if both n[j, :].sum() == 0 and n[:, j].sum() == 0
            remove_states = [j for j in range(self.K) if self.n[j, :].sum() == 0 and self.n[:, j].sum() == 0]
            if remove_states:
                keep_states = [j for j in range(self.K) if j not in remove_states]
                mapping = {old: new for new, old in enumerate(keep_states)}
                self.z = np.array([mapping[z_] for z_ in self.z if z_ in mapping])
                self.estimators = [self.estimators[j] for j in keep_states]
                self.betas = self.betas[keep_states]
                self.n = self.n[np.ix_(keep_states, keep_states)]
                self.K = len(keep_states)

            # 3.2 Update estimators for each state
            for i, estimator in enumerate(self.estimators):
                self.estimators[i] = self._update_estimator(estimator, data, i)

            # 4. Sample auxiliary variables m, w, bar_m
            m = np.zeros((self.K, self.K), dtype=int)
            for j in range(self.K):
                for k in range(self.K):
                    n_jk = self.n[j, k]
                    n = 0
                    for i in range(n_jk):
                        p = (self.alpha * self.betas[k] + self.kappa * int(j == k)) / (n + self.alpha * self.betas[k] + self.kappa * int(j == k))
                        x = np.random.rand() < p
                        if x:
                            m[j, k] += 1
                        n += 1
            # handle case where self.K is 1
            if self.K == 1:
                m[0, 0] = self.n[0, 0]

            w = np.zeros(self.K, dtype=int)
            for j in range(self.K):
                if m[j, j] > 0:
                    p = self.rho / (self.rho + self.betas[j] * (1 - self.rho))
                    w[j] = np.random.binomial(m[j, j], p)

            bar_m = np.copy(m)
            for j in range(self.K):
                for k in range(self.K):
                    if j != k:
                        bar_m[j, k] = m[j, k]
                    else:
                        bar_m[j, k] = m[j, k] - w[j]

            # 5. Sample global transition distribution beta
            bar_m_sum = bar_m.sum(axis=0)
            beta_vec = np.append(bar_m_sum, self.gamma)
            # handle case where beta_vec has 0 as element
            if np.any(beta_vec <= 0):
                beta_vec = np.maximum(beta_vec, 1e-10)
            beta_sample = np.random.dirichlet(beta_vec)
            self.betas = beta_sample[:-1]
            self.beta_tilde = beta_sample[-1]

    def fit(self, X, y = None, init_z = None, iterations=100, disp = False):
        if self.model == 'multivariate':
            data = X
        elif self.model == 'linear':
            assert y is not None and y.ndim == 1
            data = np.column_stack((X, y))
        elif self.model == 'multivariate_linear':
            assert y is not None and y.ndim == 2
            data = list(zip(X, y))

        self.init_z = init_z
        self.data = data
        if self.init_z is None:
            self._initialize(data)
        else:
            # initialize z, z-1 and z+1 with provided z
            self.init_z = np.asarray(init_z).flatten()
            self.z = self.init_z
            self.z_prev = np.roll(self.z, 1)
            self.z_next = np.roll(self.z, -1)

        for i, estimator in enumerate(self.estimators):
            self.estimators[i] = self._update_estimator(estimator, data, i)

        for it in tqdm(range(iterations), desc="Fitting HDPHMM", disable=not disp):
            self.fit_one_step(data)

    def predict(self, x):
        """
        Predict the state assignments for new data.
        This method assumes the model has been fitted.
        """
        if not hasattr(self, 'z'):
            raise ValueError("Model has not been fitted yet. Please call fit() before predict().")
        
        # Ensure x is (d)
        x = np.asarray(x)
        if x.ndim > 1:
            x = x.flatten()

        n_total = len(self.data)
        probs = []
        dists = []

        last_state = self.z[-1] if hasattr(self, 'z') else None
        trans_prob =  self.n[last_state, :]
        
        if self.model == 'multivariate':
            for t, est in enumerate(self.estimators):
                n_t = sum(self.z == t)
                prob = n_t / (n_total + self.alpha) * est.posterior(x) * trans_prob[t]
                probs.append(prob)
                mu, covs = est.predict(x)
                dists.append(multivariate_normal(mu, covs))

        elif self.model == 'linear':
            for t, est in enumerate(self.estimators):
                n_t = sum(self.z == t)
                prob = n_t / (n_total + self.alpha) * est[1].posterior(x) * trans_prob[t]
                probs.append(prob)
                dist = est[0].predict(x, return_std=False)
                dists.append(dist)

        elif self.model == 'multivariate_linear':
            for t, est in enumerate(self.estimators):
                n_t = sum(self.z == t)
                prob = n_t / (n_total + self.alpha) * est[1].posterior(x) * trans_prob[t]
                probs.append(prob)
                dist = est[0].predict(x, return_std=False)
                dists.append(dist)
    
        # state_probs must sum to 1
        probs = np.array(probs)
        probs /= np.sum(probs)
        return MixtureDistribution(probs, dists)
    
    def summary(self):
        """
        Print a summary of the model parameters.
        """
        print(f"Model type: {self.model}")
        print(f"Number of states: {self.K}")
        print(f"Transition counts (n):\n{self.n}")
        print(f"Beta weights: {self.betas}")
        print(f"Sticky parameter (rho): {self.rho}")
        for i, estimator in enumerate(self.estimators):
            print(f"Estimator {i}: {estimator.summary()}")
        if hasattr(self, 'z'):
            print(f"State assignments (z): {self.z}")
        else:
            print("Model has not been fitted yet.")