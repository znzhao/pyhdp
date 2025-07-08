import numpy as np
from tqdm import tqdm
from typing import Literal
from api.bayes import BayesianMultivariateNormalEstimator
from api.bayes import BayesianLinearRegression
from api.bayes import BayesianMultivariateLinearRegression
from api.utils import MixtureDistribution
from scipy.stats import multivariate_normal

class DirichletProcessClassifier:
    def __init__(self, alpha, weight = 1.0, kappa_0 = 1.0, nu_0 = 5.0, a_0 = 2.0, b_0 = 2.0, 
                 model: Literal['multivariate', 'linear', 'multivariate_linear'] = 'multivariate'):
        """
        Initialize the Dirichlet Process Classifier.

        Parameters:
        - alpha: Concentration parameter (float), controls the likelihood of creating new clusters.
        - step: Step size for updating cluster parameters (float, default=1).
        - H_loc: Base distribution for the location parameter (default is standard normal distribution).
        - H_scale: Base distribution for the scale parameter (default is gamma distribution).
        - model: 'multivariate', 'linear', or 'multivariate_linear'
        """
        self.alpha = alpha
        self.model = model
        self.weight = weight
        self.tables = []  # Clusters (tables)
        self.estimators = []  # Estimators for each cluster
        self.kappa_0 = kappa_0  # Prior for multivariate normal
        self.nu_0 = nu_0  # Prior for multivariate normal
        self.a_0 = a_0  # Prior for linear regression
        self.b_0 = b_0  # Prior for linear regression

    @property
    def params(self):
        """
        Get the parameters of the Dirichlet Process Classifier.

        Returns:
        - Dictionary containing the parameters.
        """
        return {
            'alpha': self.alpha,
            'weight': self.weight,
            'kappa_0': self.kappa_0,
            'nu_0': self.nu_0,
            'a_0': self.a_0,
            'b_0': self.b_0,
            'model': self.model
        }

    def _initialize(self, data):
        """Initialize the Dirichlet Process Classifier with the given data."""
        if self.init_tables is not None:
            self.tables = self.init_tables
            self.estimators = [self._init_estimators(x, y) for x, y in self.init_tables]
            return

        # Initialize clusters randomly
        for i in range(len(data)):
            if self.model == 'multivariate':
                x = data[i]
                y_val = None
            elif self.model == 'linear':
                x = data[i, :-1]
                y_val = data[i, -1]
            elif self.model == 'multivariate_linear':
                x, y_val = data[i]

            if len(self.tables) == 0:
                est = self._init_estimators(x, y_val)
                self.tables.append([(x, y_val)])
                self.estimators.append(est)
            else:
                p = [len(table) / (np.sum([len(t) for t in self.tables]) + self.alpha) for table in self.tables]
                p.append(self.alpha / (np.sum([len(t) for t in self.tables]) + self.alpha))
                choice = np.random.choice(len(self.tables) + 1, p = p)
                if choice == len(self.tables):
                    est = self._init_estimators(x, y_val)
                    self.tables.append([(x, y_val)])
                    self.estimators.append(est)
                else:
                    self.tables[choice].append((x, y_val))

        # initialize fits for each cluster
        for choice, table in enumerate(self.tables):
            if self.model == 'multivariate':
                self.estimators[choice].fit(np.array([t[0] for t in table]))
            elif self.model == 'linear':
                self.estimators[choice][0].fit(np.array([t[0] for t in table]), np.array([t[1] for t in table]))
                self.estimators[choice][1].fit(np.array([t[0] for t in table]))
            elif self.model == 'multivariate_linear':
                self.estimators[choice][0].fit(np.array([t[0] for t in table]), np.array([t[1] for t in table]))
                self.estimators[choice][1].fit(np.array([t[0] for t in table]))

    def _init_estimators(self, x, y=None):
        """Initialize the cluster estimators."""
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
            return BayesianLinearRegression(beta_0, Sigma_0, a_0, b_0), BayesianMultivariateNormalEstimator(mu_0, kappa_0, nu_0, psi_0)
        elif self.model == 'multivariate_linear':
            d = x.shape[0] if x.ndim == 1 else x.shape[1]
            m = x.shape[0] if x.ndim == 1 else x.shape[1]
            k = y.shape[0] if y is not None else 1
            
            M0 = np.zeros((m, k))
            V0 = np.eye(m)
            S0 = np.eye(k)
            nu0 = k + 2

            mu_0 = np.zeros(d)
            kappa_0 = self.kappa_0
            nu_0 = self.nu_0
            psi_0 = np.eye(d)
            return  BayesianMultivariateLinearRegression(M0, V0, S0, nu0), BayesianMultivariateNormalEstimator(mu_0, kappa_0, nu_0, psi_0)

    def _remove_data_point(self, x, y=None):
        """Remove a data point from its current cluster."""
        for i, table in enumerate(self.tables):
            for j, (x_val, y_val) in enumerate(table):
                if np.array_equal(x_val, x) and (y is None or np.array_equal(y_val, y)):
                    table.pop(j)
                    if len(table) == 0:
                        # Remove empty cluster
                        self.tables.pop(i)
                        self.estimators.pop(i)
                    return

    def _calculate_posterior(self, x, y=None):
        """Calculate the posterior probabilities for each cluster."""
        probs = []
        for i, table in enumerate(self.tables):
            if self.model == 'multivariate':
                est = self.estimators[i]
                prob = est.posterior(x)
            else:
                est = self.estimators[i]
                proby = est[0].posterior(x, y)
                probx = est[1].posterior(x)
                prob = proby ** self.weight * probx ** (1-self.weight)
            probs.append(len(table) * prob)
        # Add probability of creating a new cluster
        # Compute the probability of creating a new cluster (prior * likelihood under base measure)
        if self.model == 'multivariate':
            # Use a fresh estimator with prior parameters
            base_est = self._init_estimators(x)
            prob_new = self.alpha * base_est.posterior(x)
        elif self.model == 'linear':
            base_est = self._init_estimators(x, y)
            prob_new = self.alpha * base_est[0].posterior(x, y) * base_est[1].posterior(x)
        elif self.model == 'multivariate_linear':
            base_est = self._init_estimators(x, y)
            prob_new = self.alpha * base_est[0].posterior(x, y) * base_est[1].posterior(x)
        probs.append(prob_new)
        return np.array(probs) / sum(probs)
    
    def _update_estimator(self, estimator, table):
        """Update the estimator with the current cluster data."""
        if self.model == 'multivariate':
            x = np.array([t[0] for t in table])
            estimator = self._init_estimators(x)
            estimator.fit(x)
        else:
            x = np.array([t[0] for t in table])
            y = np.array([t[1] for t in table])
            if self.model == 'linear':
                x_val = x[-1]
                y_val = y[-1]
            else:
                x_val, y_val = table[-1]
            estimator = self._init_estimators(x_val, y_val)

            estimator[0].fit(x, y)
            estimator[1].fit(x)
        return estimator

    def fit(self, X, y=None, init_tables=None, iterations=100, disp=False):
        """
        Fit the Dirichlet Process Classifier to the data.
        Parameters:
        - X: (n, d) numpy array (features).
        - y: (optional) (n,) or (n, k) numpy array (labels, only used in linear/multivariate_linear model).
        - iterations: Number of iterations for Gibbs sampling (int, default=100).

        Returns:
        - None
        """
        if self.model == 'multivariate':
            data = X
        elif self.model == 'linear':
            assert y is not None and y.ndim == 1
            data = np.column_stack((X, y))
        elif self.model == 'multivariate_linear':
            assert y is not None and y.ndim == 2
            data = list(zip(X, y))
        self.init_tables =  init_tables
        self._initialize(data)

        # Gibbs sampling
        for iter in tqdm(range(iterations), desc="Fitting model", disable=not disp):
            self.fit_one_step(data)

    def fit_one_step(self, data):
        for i in range(len(data)):
            if self.model == 'multivariate':
                x = data[i]
                y_val = None
            elif self.model == 'linear':
                x = data[i, :-1]
                y_val = data[i, -1]
            elif self.model == 'multivariate_linear':
                x, y_val = data[i]
            self._remove_data_point(x, y_val)
            posteriors = self._calculate_posterior(x, y_val)
            choice = np.random.choice(len(posteriors), p=posteriors)
            if choice == len(self.tables):
                est = self._init_estimators(x, y_val)
                self.tables.append([(x, y_val)])
                self.estimators.append(est)
            else:
                self.tables[choice].append((x, y_val))

        for choice in range(len(self.tables)):
            self.estimators[choice] = self._update_estimator(self.estimators[choice], self.tables[choice])


    def classify(self, x, y=None):
        """
        Classify a new observation x using the current clusters.
        Parameters:
        - x: (d,) numpy array (a new data point)
        - y: (optional) label for the new data point (only used in linear/multivariate_linear model).

        Returns:
        - choice: Index of the cluster to which the observation is assigned.
        """
        if self.model == 'multivariate':
            x = np.asarray(x)
            y_val = None
        elif self.model == 'linear':
            x = np.asarray(x)
            y_val = x[-1]
        elif self.model == 'multivariate_linear':
            x = np.asarray(x)
            y_val = np.asarray(y)
        posteriors = self._calculate_posterior(x, y_val)
        choice = np.random.choice(len(posteriors), p=posteriors)
        return choice

    def get_clusters(self):
        """
        Get the current clusters (tables).

        Returns:
        - List of clusters (each cluster is a list of observations).
        """
        return self.tables, self.estimators
    
    def predict(self, X):
        """
        Predict the cluster for each observation in X.

        For multivariate model, some observations may have NaN values, and we will predict the distribution of the NaN values.
        For linear and multivariate_linear models, we will predict the distribution of y based on the features.

        Parameters:
        - X: (n, d) numpy array (features).

        Returns:
        - preds: list of predictive distributions or values for each observation.
        """
        # Ensure X is (n, d)
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_total = sum(len(table) for table in self.tables)
        mix_dists = []
        for x in X:
            probs = []
            dists = []
            if self.model == 'multivariate':
                x = np.asarray(x)
                if x.ndim > 1:
                    x = x.flatten()
                for t, (table, est) in enumerate(zip(self.tables, self.estimators)):
                    n_t = len(table)
                    prob = n_t / (n_total + self.alpha) * est.posterior(x)

                    probs.append(prob)
                    mu, covs = est.predict(x)
                    dists.append(multivariate_normal(mu.flatten(), covs[0], allow_singular=True))

            else:
                x = np.asarray(x)
                if x.ndim > 1:
                    x = x.flatten()
                for t, (table, est) in enumerate(zip(self.tables, self.estimators)):
                    n_t = len(table)
                    prob = n_t / (n_total + self.alpha) * est[1].posterior(x)
                    probs.append(prob)
                    dist = est[0].predict(x, return_std=False)
                    dists.append(dist)
        
            # state_probs must sum to 1
            probs = np.array(probs)
            probs /= np.sum(probs)
            mix_dists.append(MixtureDistribution(probs, dists))
        return mix_dists
    
    def summary(self):
        """
        Summarize the mean of the cluster parameters.
        """
        summaries = []
        for i, table in enumerate(self.tables):
            if self.model == 'multivariate':
                est = self.estimators[i]
                summaries.append((est.mu, est.kappa, est.nu, est.psi))
            elif self.model == 'linear':
                est = self.estimators[i]
                summaries.append((est.beta, est.Sigma, est.a, est.b))
            elif self.model == 'multivariate_linear':
                est = self.estimators[i]
                summaries.append((est.Mn, est.Vn, est.Sn, est.nun))
        return summaries

