import logging
import numpy as np
from tqdm import tqdm
import matplotlib.animation as animation
from matplotlib import pyplot
from api.dp import DirichletProcessClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_multivariate():
    """
    Test the Dirichlet Process Classifier with multivariate data.
    This function simulates data with two states, fits the model, and visualizes the clustering process.
    """

    logging.info("Starting multivariate test...")

    # Simulate data with 2 states
    np.random.seed(42)
    X1 = np.random.multivariate_normal(mean=[1, 7], cov=np.array([[2, -0.9], [-0.9, 1]]), size=50)
    X2 = np.random.multivariate_normal(mean=[8, 2], cov=np.array([[4, 2], [2, 2]]), size=50)
    X = np.vstack((X1, X2))
    states = np.array([0] * 50 + [1] * 50)
    indices = np.random.permutation(len(X))
    X = X[indices]
    states = states[indices]

    fig, ax = pyplot.subplots(figsize=(6, 5))
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

    # Run clustering and store cluster assignments at each iteration
    model = DirichletProcessClassifier(alpha=1, model='multivariate')
    data_snapshots = []
    X_anim = X.copy()
    model.fit(X_anim, iterations=1)  # initialize
    for _ in tqdm(range(100), desc="Fitting model"):
        model.fit_one_step(X_anim)
        clusters, _ = model.get_clusters()
        labels = np.zeros(len(X_anim), dtype=int)
        for idx, cluster in enumerate(clusters):
            for point in cluster:
                # Find the index of this point in X_anim
                matches = np.where((X_anim == point[0]).all(axis=1))[0]
                for m in matches:
                    labels[m] = idx
        data_snapshots.append(labels.copy())

    def update(frame):
        ax.clear()
        labels = data_snapshots[frame]
        for i in range(len(np.unique(labels))):
            mask = labels == i
            ax.scatter(X_anim[mask, 0], X_anim[mask, 1], color=cluster_colors[i % len(cluster_colors)], s=20, alpha=0.7, label=f"Cluster {i+1}")
        ax.set_title(f"Clustering Animation (Iteration {frame+1})")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend(loc='upper right')

    ani = animation.FuncAnimation(fig, update, frames=len(data_snapshots), interval=50, repeat=False)
    pyplot.show()

    clusters, _ = model.get_clusters()
    pyplot.figure(figsize=(12, 6))
    pyplot.subplot(1, 2, 1)
    pyplot.scatter(X[:, 0], X[:, 1], c=states, cmap='viridis', s=20, alpha=0.7)
    pyplot.title("True Data Distribution")
    pyplot.xlabel("Feature 1")
    pyplot.ylabel("Feature 2")
    pyplot.subplot(1, 2, 2)
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    for i, cluster in enumerate(clusters):
        cluster_data = np.array([point[0] for point in cluster])
        pyplot.scatter(cluster_data[:, 0], cluster_data[:, 1], color=cluster_colors[i % len(cluster_colors)], s=20, alpha=0.7, label=f"Cluster {i+1}")
    pyplot.title("Estimated Clusters")
    pyplot.xlabel("Feature 1")
    pyplot.ylabel("Feature 2")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()

    # test for predict one x data point
    x_test = np.array([[2, np.nan]])
    preds = model.predict(x_test)
    pred = preds[0]  # Get the first prediction
    # plot the predicted distribution pdf
    x_vals = np.linspace(-5, 10, 100)
    pdf_vals = pred.pdf(x_vals)
    pyplot.figure(figsize=(8, 4))
    pyplot.plot(x_vals, pdf_vals, label="Predicted PDF", color='orange')
    pyplot.title("Predicted Distribution PDF")
    pyplot.xlabel("predicted value")
    pyplot.ylabel("PDF")
    max_x = pred.max_idx(min_val=x_vals[0], max_val=x_vals[-1], num_points=len(x_vals))
    pyplot.axvline(max_x, color="red", linestyle="--", label=f"max_idx={max_x:.2f}")
    pyplot.legend()
    pyplot.show()

    logging.info("Multivariate test completed successfully.")

def test_linear():
    """
    Test the Dirichlet Process Classifier with linear data.
    This function simulates data with two states, fits the model, and visualizes the clustering
    process with linear regression.
    """
    logging.info("Starting linear test...")
    
    # Simulate data with 2 states
    np.random.seed(60)
    X1 = np.random.multivariate_normal(mean=[3, 3], cov=np.array([[4, -3.5], [-3.5, 4]]), size=50)
    X2 = np.random.multivariate_normal(mean=[2, 1], cov=np.array([[4, 3.5], [3.5, 4]]), size=50)
    X = np.vstack((X1, X2))
    states = np.array([0] * 50 + [1] * 50)
    indices = np.random.permutation(len(X))
    X = X[indices]
    states = states[indices]

    model_linear = DirichletProcessClassifier(alpha=0.1, model='linear')
    X_input = np.hstack([X[:, 0][:, np.newaxis], np.ones(shape=(X.shape[0], 1))])
    y_input = X[:, 1]
    data_input = np.hstack([X_input, y_input[:, np.newaxis]])
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    fig, ax = pyplot.subplots(figsize=(6, 5))
    data_snapshots = []

    model_linear.fit(X_input, y=y_input, iterations=1)  # initialize
    for _ in tqdm(range(100), desc="Fitting model"):
        model_linear.fit_one_step(data_input)
        clusters, estimators = model_linear.get_clusters()
        labels = np.zeros(len(X_input), dtype=int)
        for idx, cluster in enumerate(clusters):
            for point in cluster:
                matches = np.where((X_input == point[0]).all(axis=1))[0]
                for m in matches:
                    labels[m] = idx
        data_snapshots.append(labels.copy())

    def update(frame):
        ax.clear()
        labels = data_snapshots[frame]
        for i in range(len(np.unique(labels))):
            mask = labels == i
            ax.scatter(X_input[mask, 0], y_input[mask], color=cluster_colors[i % len(cluster_colors)], s=20, alpha=0.7, label=f"Cluster {i+1}")
            # Plot regression line for each cluster
            if len(X_input[mask, 0]) > 1:
                x_vals = np.linspace(X_input[mask, 0].min(), X_input[mask, 0].max(), 100)
                beta, scale = estimators[i][0].estimate()
                y_vals = beta[1] + beta[0] * x_vals
                ax.plot(x_vals, y_vals, color=cluster_colors[i % len(cluster_colors)], linestyle='--')

        ax.set_title(f"Clustering Animation (Iteration {frame+1})")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend(loc='upper right')

    ani = animation.FuncAnimation(fig, update, frames=len(data_snapshots), interval=50, repeat=False)
    pyplot.show()

    clusters_linear, estimators = model_linear.get_clusters()
    pyplot.figure(figsize=(12, 6))
    pyplot.subplot(1, 2, 1)
    pyplot.scatter(X[:, 0], X[:, 1], c=states, cmap='viridis', s=20, alpha=0.7)
    pyplot.title("True Data Distribution")
    pyplot.xlabel("Feature 1")
    pyplot.ylabel("Feature 2")
    pyplot.subplot(1, 2, 2)
    for i, (cluster, estimator) in enumerate(zip(clusters_linear, estimators)):
        cluster_data_x = np.array([point[0][0] for point in cluster])
        cluster_data_y = np.array([point[1] for point in cluster])
        pyplot.scatter(cluster_data_x, cluster_data_y, color=cluster_colors[i % len(cluster_colors)], s=20, alpha=0.7, label=f"Cluster {i+1}")
        x_vals = np.linspace(cluster_data_x.min(), cluster_data_x.max(), 100)
        beta, scale = estimator[0].estimate()
        y_vals = beta[1] + beta[0] * x_vals
        pyplot.plot(x_vals, y_vals, color=cluster_colors[i % len(cluster_colors)], linestyle='--', label=f"Regression Line {i+1}")
    pyplot.title("Linear Regression Results")
    pyplot.xlabel("Feature 1")
    pyplot.ylabel("Feature 2")
    pyplot.legend()
    pyplot.tight_layout()
    pyplot.show()

    # test for predict one x data point
    x_test = np.array([2, 1])
    preds = model_linear.predict(x_test)
    pred = preds[0]  # Get the first prediction
    # plot the predicted distribution pdf
    x_vals = np.linspace(-5, 10, 100)
    pdf_vals = pred.pdf(x_vals)
    pyplot.figure(figsize=(8, 4))
    pyplot.plot(x_vals, pdf_vals, label="Predicted PDF", color='orange')
    pyplot.title("Predicted Distribution PDF")
    pyplot.xlabel("predicted value")
    pyplot.ylabel("PDF")
    max_x = pred.max_idx(min_val=x_vals[0], max_val=x_vals[-1], num_points=len(x_vals))
    pyplot.axvline(max_x, color="red", linestyle="--", label=f"max_idx={max_x:.2f}")
    pyplot.legend()
    pyplot.show()

    logging.info("Linear test completed successfully.")

def test_multivariate_linear():
    """
    Test the Dirichlet Process Classifier with multivariate linear data.
    This function simulates data with two states, fits the model, and visualizes the clustering
    process with multivariate linear regression.
    """

    logging.info("Starting multivariate linear test...")

    # Simulate data with 2 states
    np.random.seed(60)
    X1 = np.random.multivariate_normal(mean=[4, 3, 3], cov=np.array([[4, -3.5, 0.5], [-3.5, 4, 0.2], [0.5, 0.2, 2]]), size=50)
    X2 = np.random.multivariate_normal(mean=[2, 1, -1], cov=np.array([[4, 3.5, 0.3], [3.5, 4, 0.1], [0.3, 0.1, 1]]), size=50)
    X = np.vstack((X1, X2))
    states = np.array([0] * 50 + [1] * 50)
    indices = np.random.permutation(len(X))
    X = X[indices]
    states = states[indices]

    model_multilinear = DirichletProcessClassifier(alpha=20, model='multivariate_linear')
    X_input = np.hstack([X[:, 0][:, np.newaxis], np.ones(shape=(X.shape[0], 1))])
    y_input = X[:, 1:]
    data_input = list(zip(X_input, y_input))
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    import matplotlib.pyplot as pyplot
    import matplotlib.animation as animation

    fig, ax = pyplot.subplots(figsize=(6, 5))
    data_snapshots = []

    model_multilinear.fit(X_input, y=y_input, iterations=1)  # initialize
    for _ in tqdm(range(100), desc="Fitting model"):
        model_multilinear.fit_one_step(data_input)
        clusters, estimators = model_multilinear.get_clusters()
        labels = np.zeros(len(X_input), dtype=int)
        for idx, cluster in enumerate(clusters):
            for point in cluster:
                matches = np.where((X_input == point[0]).all(axis=1))[0]
                for m in matches:
                    labels[m] = idx
        data_snapshots.append((labels.copy(), estimators.copy()))

    def update(frame):
        fig.set_size_inches(12, 5)  # Set the desired figsize here
        ax.clear()
        labels, estimators = data_snapshots[frame]
        # Create a 1x2 subplot: (F1 vs F2) and (F1 vs F3)
        ax1 = ax
        ax1.set_position([0.07, 0.15, 0.4, 0.75])  # left, bottom, width, height
        ax2 = fig.add_axes([0.55, 0.15, 0.4, 0.75])

        for i in range(len(np.unique(labels))):
            mask = labels == i
            # F1 vs F2
            ax1.scatter(X_input[mask, 0], y_input[mask, 0], color=cluster_colors[i % len(cluster_colors)], s=20, alpha=0.7, label=f"Cluster {i+1}")
            if len(X_input[mask, 0]) > 1:
                x_vals = np.linspace(X_input[mask, 0].min(), X_input[mask, 0].max(), 100)
                beta_mean, Sigma_mean = estimators[i][0].estimate()
                y_vals = beta_mean[0][0] * x_vals + beta_mean[1][0]
                ax1.plot(x_vals, y_vals, color=cluster_colors[i % len(cluster_colors)], linestyle='--')
            # F1 vs F3
            ax2.scatter(X_input[mask, 0], y_input[mask, 1], color=cluster_colors[i % len(cluster_colors)], s=20, alpha=0.7, label=f"Cluster {i+1}")
            if len(X_input[mask, 0]) > 1:
                x_vals = np.linspace(X_input[mask, 0].min(), X_input[mask, 0].max(), 100)
                beta_mean, Sigma_mean = estimators[i][0].estimate()
                # For F3, use the second column of beta_mean if available
                if beta_mean.shape[1] > 1:
                    y_vals = beta_mean[0][1] * x_vals + beta_mean[1][1]
                    ax2.plot(x_vals, y_vals, color=cluster_colors[i % len(cluster_colors)], linestyle='--')

        ax1.set_title(f"F1 vs F2 (Iteration {frame+1})")
        ax1.set_xlabel("Feature 1")
        ax1.set_ylabel("Feature 2")
        ax1.legend(loc='upper right')

        ax2.set_title(f"F1 vs F3 (Iteration {frame+1})")
        ax2.set_xlabel("Feature 1")
        ax2.set_ylabel("Feature 3")
        ax2.legend(loc='upper right')

    ani = animation.FuncAnimation(fig, update, frames=len(data_snapshots), interval=100, repeat=False)
    pyplot.show()

    clusters_multilinear, multi_estimators = model_multilinear.get_clusters()
    pyplot.figure(figsize=(14, 10))
    # 2x2 subplots: 
    # (1,1) True F1 vs F2, (1,2) Fitted F1 vs F2
    # (2,1) True F1 vs F3, (2,2) Fitted F1 vs F3

    # First subplot: True Feature 1 vs Feature 2
    ax1 = pyplot.subplot(2, 2, 1)
    sc1 = ax1.scatter(X[:, 0], X[:, 1], c=states, cmap='viridis', s=20, alpha=0.7)
    ax1.set_title("True Data (F1 vs F2)")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")

    # Second subplot: Estimated clusters (Feature 1 vs Feature 2)
    ax2 = pyplot.subplot(2, 2, 2)
    for i, (cluster, multi_estimator) in enumerate(zip(clusters_multilinear, multi_estimators)):
        cluster_data_x = np.array([point[0][0] for point in cluster])
        cluster_data_y = np.array([point[1][0] for point in cluster])
        ax2.scatter(cluster_data_x, cluster_data_y, color=cluster_colors[i % len(cluster_colors)], s=20, alpha=0.7, label=f"Cluster {i+1}")
        if len(cluster_data_x) > 1:
            x_vals = np.linspace(cluster_data_x.min(), cluster_data_x.max(), 100)
            beta_mean, Sigma_mean = multi_estimator[0].estimate()
            y_vals = beta_mean[0][0] * x_vals + beta_mean[1][0]
            ax2.plot(x_vals, y_vals, color=cluster_colors[i % len(cluster_colors)], linestyle='--', label=f"Regression Line {i+1}")
    ax2.set_title("Estimated Clusters (F1 vs F2)")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.legend()

    # Third subplot: True Feature 1 vs Feature 3
    ax3 = pyplot.subplot(2, 2, 3)
    sc3 = ax3.scatter(X[:, 0], X[:, 2], c=states, cmap='viridis', s=20, alpha=0.7)
    ax3.set_title("True Data (F1 vs F3)")
    ax3.set_xlabel("Feature 1")
    ax3.set_ylabel("Feature 3")

    # Fourth subplot: Estimated clusters (Feature 1 vs Feature 3)
    ax4 = pyplot.subplot(2, 2, 4)
    for i, (cluster, _) in enumerate(zip(clusters_multilinear, multi_estimators)):
        cluster_data_x = np.array([point[0][0] for point in cluster])
        # Find the index in X for each cluster point to get F3
        cluster_data_z = np.array([X[np.where((X_input == point[0]).all(axis=1))[0][0], 2] for point in cluster])
        ax4.scatter(cluster_data_x, cluster_data_z, color=cluster_colors[i % len(cluster_colors)], s=20, alpha=0.7, label=f"Cluster {i+1}")
        if len(cluster_data_x) > 1:
            x_vals = np.linspace(cluster_data_x.min(), cluster_data_x.max(), 100)
            beta_mean, Sigma_mean = multi_estimators[i][0].estimate()
            y_vals = beta_mean[0][1] * x_vals + beta_mean[1][1]
            ax4.plot(x_vals, y_vals, color=cluster_colors[i % len(cluster_colors)], linestyle='--', label=f"Regression Line {i+1}")
    ax4.set_title("Estimated Clusters (F1 vs F3)")
    ax4.set_xlabel("Feature 1")
    ax4.set_ylabel("Feature 3")
    ax4.legend()
    pyplot.tight_layout()
    pyplot.show()

    # test for predict one x data point
    x_test = np.array([2, 1])
    preds = model_multilinear.predict(x_test)
    pred = preds[0]  # Get the first prediction
    # Plot the predicted joint distribution PDF for Feature 2 and Feature 3 using a meshgrid and 3D plot
    f2_range = np.linspace(-5, 10, 100)
    f3_range = np.linspace(-5, 10, 100)
    F2, F3 = np.meshgrid(f2_range, f3_range)
    pos = np.dstack((F2, F3))
    # pred.pdf expects shape (N, 2)
    pdf_vals = pred.pdf(np.column_stack([F2.ravel(), F3.ravel()])).reshape(F2.shape)

    fig = pyplot.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(F2, F3, pdf_vals, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_title("Predicted Joint PDF (Feature 2 vs Feature 3)")
    ax.set_xlabel("Feature 2")
    ax.set_ylabel("Feature 3")
    ax.set_zlabel("PDF")
    fig.colorbar(surf, shrink=0.5, aspect=10, label="PDF Value")
    pyplot.show()

def run_tests():
    """
    Run all tests in the module.
    This function is used to execute the tests sequentially.
    """
    
    logging.info("Running tests...")

    test_multivariate()
    test_linear()
    test_multivariate_linear()
    
    logging.info("All tests completed successfully.")

if __name__ == "__main__":
    run_tests()