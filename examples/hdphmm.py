import logging
import numpy as np
from matplotlib import pyplot
from api.hdphmm import StickyHDPHMM
from matplotlib import pyplot
import matplotlib.animation as animation


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_multivariate():
    # Example usage
    np.random.seed(52)
    # Generate data that alternates between two states
    mean1 = np.array([7.0, 5.0, 8.0])
    mean2 = np.array([3.0, 7.0, 2.0])
    
    cov1 = 2 * np.array([[1.0, 0.9, 0.9],
                     [0.9, 1.0, 0.9],
                     [0.9, 0.9, 1.0]])
    
    cov2 = 2 * np.array([[1.0, -0.9, -0.9],
                     [-0.9, 1.0, 0.9],
                     [-0.9, 0.9, 1.0]])
    
    X = np.zeros((100, 3))
    for i in range(100):
        if int(i/10) % 2 == 0:
            X[i] = np.random.multivariate_normal(mean1, cov1)
        else:
            X[i] = np.random.multivariate_normal(mean2, cov2)

    model = StickyHDPHMM(ups=0.75, gamma=0.5, rho=0.7, model='multivariate')
    # Fit the model to the data
    logger.info("Fitting HDP-HMM model to data...")
    model.fit(X, iterations=1)

    steps = 50
    fig, ax = pyplot.subplots(figsize=(6, 5))

    scat = ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5)
    title = ax.set_title("HDP-HMM Fitting Progress")

    def update(step):
        model.fit_one_step(X)
        colors = model.z if hasattr(model, 'z') else np.zeros(X.shape[0])
        scat.set_array(colors)
        title.set_text(f"HDP-HMM Fitting Progress - Step {step+1}")
        return scat, title

    ani = animation.FuncAnimation(
        fig, update, frames=steps, interval=100, blit=False, repeat=False
    )
    pyplot.show()

    # report status of the class. log all attributes
    logger.info(f"Model type: {model.model}")
    logger.info(f"Number of states: {model.K}")
    logger.info(f"Transition counts (n): \n{model.n}")
    logger.info(f"Beta weights: {model.betas}")
    logger.info(f"Sticky parameter (rho): {model.rho}")
    for i, estimator in enumerate(model.estimators):
        logger.info(f"Estimator {i}: {estimator}")

    # Check if the model has been fitted successfully
    if hasattr(model, 'z'):
        logger.info(f"State assignments (z): \n{model.z}")
    else:
        logger.warning("Model has not been fitted yet.")

    logger.info("Model fitted successfully.")
    
    # Plot true clusters (alternating every 10 points)
    pyplot.figure(figsize=(12, 5))
    pyplot.subplot(1, 2, 1)
    for i in range(0, 100, 10):
        color = 'blue' if (i // 10) % 2 == 0 else 'red'
        pyplot.scatter(X[i:i+10, 0], X[i:i+10, 1], color=color, alpha=0.5, label=f'True cluster {1 + ((i // 10) % 2)}' if i < 20 else "")
    pyplot.title("True Data Clusters (first 2 dims)")
    pyplot.xlabel("X[:, 0]")
    pyplot.ylabel("X[:, 1]")
    pyplot.legend()

    # Plot estimated clusters
    pyplot.subplot(1, 2, 2)
    for k in range(model.K):
        idx = np.where(model.z == k)[0]
        pyplot.scatter(X[idx, 0], X[idx, 1], alpha=0.5, label=f'Estimated cluster {k}')
    pyplot.title("Estimated Data Clusters (first 2 dims)")
    pyplot.xlabel("X[:, 0]")
    pyplot.ylabel("X[:, 1]")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()

    # Time series plot of cluster assignments
    pyplot.figure(figsize=(12, 3))
    pyplot.plot(model.z, label='Estimated cluster')
    pyplot.plot([(i // 10) % 2 for i in range(100)], '--', label='True cluster', alpha=0.7)
    pyplot.title("Cluster Assignments Over Time")
    pyplot.xlabel("Time")
    pyplot.ylabel("Cluster")
    pyplot.legend()
    pyplot.show()

    # test for predict one x data point
    x_test = np.array([-2, 2, np.nan])
    pred = model.predict(x_test)
    # plot the predicted distribution pdf
    x_vals = np.linspace(-5, 10, 100)
    pdf_vals = pred.pdf(x_vals)
    pyplot.figure(figsize=(8, 4))
    pyplot.plot(x_vals, pdf_vals, label="Predicted PDF", color='orange')
    pyplot.title("Predicted Distribution PDF")
    pyplot.xlabel("x")
    pyplot.ylabel("PDF")
    max_x = pred.max_idx(min_val=x_vals[0], max_val=x_vals[-1], num_points=len(x_vals))
    pyplot.axvline(max_x, color="red", linestyle="--", label=f"max_idx={max_x:.2f}")
    pyplot.legend()
    pyplot.show()

def test_linear():
    """
    Test the Dirichlet Process Classifier with linear data.
    This function simulates data with two states, fits the model, and visualizes the clustering
    process with linear regression.
    """
    logging.info("Starting linear test...")

    # Simulate data with 2 states
    np.random.seed(60)
    X = np.zeros((100, 2))
    states = np.zeros(100, dtype=int)  # To keep track of the true state for visualization
    for i in range(100):
        if int(i/10) % 2 == 0:
            X[i] = np.random.multivariate_normal(mean=[3, 3], cov=np.array([[4, -3.5], [-3.5, 4]]))
            states[i] = 0
        else:
            X[i] = np.random.multivariate_normal(mean=[2, 1], cov=np.array([[6, 5.5], [5.5, 6]]))
            states[i] = 1

    model = StickyHDPHMM(ups=0.75, gamma=0.5, rho=0.7, model='linear')
    X_input = np.hstack([X[:, 0][:, np.newaxis], np.ones(shape=(X.shape[0], 1))])
    y_input = X[:, 1]
    data_input = np.hstack([X_input, y_input[:, np.newaxis]])
    steps = 50
    fig, ax = pyplot.subplots(figsize=(6, 5))

    scat = ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5)
    title = ax.set_title("HDP-HMM Fitting Progress")
    model.fit(X_input, y_input, iterations=1)

    def update(step):
        model.fit_one_step(data_input)
        colors = model.z if hasattr(model, 'z') else np.zeros(X.shape[0])
        scat.set_array(colors)
        title.set_text(f"HDP-HMM Fitting Progress - Step {step+1}")
        return scat, title

    ani = animation.FuncAnimation(
        fig, update, frames=steps, interval=100, blit=False, repeat=False
    )
    pyplot.show()

    # Plot true clusters (alternating every 10 points)
    pyplot.figure(figsize=(12, 5))
    pyplot.subplot(1, 2, 1)
    for i in range(0, 100, 10):
        color = 'blue' if (i // 10) % 2 == 0 else 'red'
        pyplot.scatter(X[i:i+10, 0], X[i:i+10, 1], color=color, alpha=0.5, label=f'True cluster {1 + ((i // 10) % 2)}' if i < 20 else "")
    pyplot.title("True Data Clusters (first 2 dims)")
    pyplot.xlabel("X[:, 0]")
    pyplot.ylabel("X[:, 1]")
    pyplot.legend()

    # Plot estimated clusters
    pyplot.subplot(1, 2, 2)
    for k in range(model.K):
        idx = np.where(model.z == k)[0]
        pyplot.scatter(X[idx, 0], X[idx, 1], alpha=0.5, label=f'Estimated cluster {k}')
    pyplot.title("Estimated Data Clusters (first 2 dims)")
    pyplot.xlabel("X[:, 0]")
    pyplot.ylabel("X[:, 1]")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()

    # Time series plot of cluster assignments
    pyplot.figure(figsize=(12, 3))
    if hasattr(model, 'z'):
        pyplot.plot(model.z, label='Estimated cluster')
    pyplot.plot([(i // 10) % 2 for i in range(100)], '--', label='True cluster', alpha=0.7)
    pyplot.title("Cluster Assignments Over Time")
    pyplot.xlabel("Time")
    pyplot.ylabel("Cluster")
    pyplot.legend()
    pyplot.show()

    # test for predict one x data point
    x_test = np.array([10, 1])
    pred = model.predict(x_test)
    # plot the predicted distribution pdf
    x_vals = np.linspace(-5, 10, 100)
    pdf_vals = pred.pdf(x_vals)
    pyplot.figure(figsize=(8, 4))
    pyplot.plot(x_vals, pdf_vals, label="Predicted PDF", color='orange')
    pyplot.title("Predicted Distribution PDF")
    pyplot.xlabel("x")
    pyplot.ylabel("PDF")
    max_x = pred.max_idx(min_val=x_vals[0], max_val=x_vals[-1], num_points=len(x_vals))
    pyplot.axvline(max_x, color="red", linestyle="--", label=f"max_idx={max_x:.2f}")
    pyplot.legend()
    pyplot.show()

    logging.info("Linear test completed successfully.")

def test_multilinear():
        # Example usage
    np.random.seed(52)
    # Generate data that alternates between two states
    mean1 = np.array([7.0, 5.0, 8.0])
    mean2 = np.array([3.0, 7.0, 2.0])
    
    cov1 = 2 * np.array([[1.0, 0.9, 0.9],
                     [0.9, 1.0, 0.9],
                     [0.9, 0.9, 1.0]])
    
    cov2 = 2 * np.array([[1.0, -0.9, -0.9],
                     [-0.9, 1.0, 0.9],
                     [-0.9, 0.9, 1.0]])
    
    X = np.zeros((100, 3))
    for i in range(100):
        if int(i/10) % 2 == 0:
            X[i] = np.random.multivariate_normal(mean1, cov1)
        else:
            X[i] = np.random.multivariate_normal(mean2, cov2)

    model = StickyHDPHMM(ups=0.75, gamma=0.5, rho=0.7, model='multivariate_linear')

    X_input = np.hstack([X[:, 0][:, np.newaxis], np.ones(shape=(X.shape[0], 1))])
    y_input = X[:, 1:]
    data_input = list(zip(X_input, y_input))
    steps = 50

    # Fit the model to the data
    logger.info("Fitting HDP-HMM model to data...")
    model.fit(X_input, y_input, iterations=1)

    steps = 50
    fig, ax = pyplot.subplots(figsize=(6, 5))

    scat = ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5)
    title = ax.set_title("HDP-HMM Fitting Progress")

    def update(step):
        model.fit_one_step(data_input)
        colors = model.z if hasattr(model, 'z') else np.zeros(X.shape[0])
        scat.set_array(colors)
        title.set_text(f"HDP-HMM Fitting Progress - Step {step+1}")
        return scat, title

    ani = animation.FuncAnimation(
        fig, update, frames=steps, interval=100, blit=False, repeat=False
    )
    pyplot.show()

    # report status of the class. log all attributes
    logger.info(f"Model type: {model.model}")
    logger.info(f"Number of states: {model.K}")
    logger.info(f"Transition counts (n): \n{model.n}")
    logger.info(f"Beta weights: {model.betas}")
    logger.info(f"Sticky parameter (rho): {model.rho}")
    for i, estimator in enumerate(model.estimators):
        logger.info(f"Estimator {i}: {estimator}")

    # Check if the model has been fitted successfully
    if hasattr(model, 'z'):
        logger.info(f"State assignments (z): \n{model.z}")
    else:
        logger.warning("Model has not been fitted yet.")

    logger.info("Model fitted successfully.")
    
    # Plot true clusters (alternating every 10 points)
    pyplot.figure(figsize=(12, 5))
    pyplot.subplot(1, 2, 1)
    for i in range(0, 100, 10):
        color = 'blue' if (i // 10) % 2 == 0 else 'red'
        pyplot.scatter(X[i:i+10, 0], X[i:i+10, 1], color=color, alpha=0.5, label=f'True cluster {1 + ((i // 10) % 2)}' if i < 20 else "")
    pyplot.title("True Data Clusters (first 2 dims)")
    pyplot.xlabel("X[:, 0]")
    pyplot.ylabel("X[:, 1]")
    pyplot.legend()

    # Plot estimated clusters
    pyplot.subplot(1, 2, 2)
    for k in range(model.K):
        idx = np.where(model.z == k)[0]
        pyplot.scatter(X[idx, 0], X[idx, 1], alpha=0.5, label=f'Estimated cluster {k}')
    pyplot.title("Estimated Data Clusters (first 2 dims)")
    pyplot.xlabel("X[:, 0]")
    pyplot.ylabel("X[:, 1]")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()

    # Time series plot of cluster assignments
    pyplot.figure(figsize=(12, 3))
    pyplot.plot(model.z, label='Estimated cluster')
    pyplot.plot([(i // 10) % 2 for i in range(100)], '--', label='True cluster', alpha=0.7)
    pyplot.title("Cluster Assignments Over Time")
    pyplot.xlabel("Time")
    pyplot.ylabel("Cluster")
    pyplot.legend()
    pyplot.show()

    # test for predict one x data point
    x_test = np.array([4, 1])
    pred = model.predict(x_test)
    # plot the predicted distribution pdf
    x_vals = np.linspace(-5, 10, 100)
    pdf_vals = pred.pdf(x_vals)
    pyplot.figure(figsize=(8, 4))
    pyplot.plot(x_vals, pdf_vals, label="Predicted PDF", color='orange')
    pyplot.title("Predicted Distribution PDF")
    pyplot.xlabel("x")
    pyplot.ylabel("PDF")
    max_x = pred.max_idx(min_val=x_vals[0], max_val=x_vals[-1], num_points=len(x_vals))
    pyplot.axvline(max_x, color="red", linestyle="--", label=f"max_idx={max_x:.2f}")
    pyplot.legend()
    pyplot.show()


def run_tests():
    """
    Run all tests for the HDP-HMM model.
    """
    logger.info("Starting HDP-HMM tests...")
    test_multivariate()
    test_linear()
    test_multilinear()
    logger.info("HDP-HMM tests completed successfully.")

if __name__ == "__main__":
    run_tests()