import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from src.simple_linear_regression import SimpleLinearRegression
from src.model_selection import ModelSelection
import sys

# Ensure src directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Create 'data' directory if it doesn't exist
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
os.makedirs(data_dir, exist_ok=True)

# Generate random collinear data
np.random.seed(0)  # For reproducibility
n_samples = 2000
n_features = 15

# Generate the first feature randomly between 0 and 10
X1 = np.random.rand(n_samples) * 10

# Generate other features as linear combinations of X1 with noise
X = np.column_stack([X1] + [2.5 * X1 + np.random.randn(n_samples) * 0.1 for _ in range(1, n_features)])

# Generate the target variable y based on a linear combination of the features with noise
coefficients = np.random.rand(n_features)  # Random coefficients for each feature
y = X.dot(coefficients) + np.random.randn(n_samples) * 2  # Linear relation with noise

# Save generated data to CSV
data_df = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(n_features)])
data_df['y'] = y
data_df.to_csv(os.path.join(data_dir, "collinear_dataset_15_features.csv"), index=False)

# Generate heatmap for collinearity between features
correlation_matrix = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(n_features)]).corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(data_dir, "feature_correlation_heatmap.png"))
plt.show()

# Initialize model and model selection
model = SimpleLinearRegression()
selector = ModelSelection(model, X, y)

# Run k-fold cross-validation
k_fold_results = selector.k_fold_cross_validation(k=10)
print("\nK-Fold Cross-Validation Results:")
print(f"Mean R^2: {k_fold_results['mean_r2']:.4f}")
print(f"Mean MAE: {k_fold_results['mean_mae']:.4f}")
print(f"Mean MSE: {k_fold_results['mean_mse']:.4f}")

# Display scores per fold
for i, (r2, mae, mse) in enumerate(
        zip(k_fold_results['r2_scores'], k_fold_results['mae_scores'], k_fold_results['mse_scores']), 1):
    print(f"Fold {i}: R^2 = {r2:.4f}, MAE = {mae:.4f}, MSE = {mse:.4f}")

# Save k-fold results to CSV
k_fold_df = pd.DataFrame({
    "Fold R^2": k_fold_results['r2_scores'],
    "Fold MAE": k_fold_results['mae_scores'],
    "Fold MSE": k_fold_results['mse_scores']
})
k_fold_df.to_csv(os.path.join(data_dir, "k_fold_results.csv"), index=False)

# Run bootstrapping
bootstrap_results = selector.bootstrap(n_iterations=100)
print("\nBootstrapping Results:")
print(f"Mean R^2: {bootstrap_results['mean_r2']:.4f}")
print(f"Mean MAE: {bootstrap_results['mean_mae']:.4f}")
print(f"Mean MSE: {bootstrap_results['mean_mse']:.4f}")

# Display scores per iteration
for i, (r2, mae, mse) in enumerate(
        zip(bootstrap_results['r2_scores'], bootstrap_results['mae_scores'], bootstrap_results['mse_scores']), 1):
    print(f"Iteration {i}: R^2 = {r2:.4f}, MAE = {mae:.4f}, MSE = {mse:.4f}")

# Save bootstrapping results to CSV
bootstrap_df = pd.DataFrame({
    "Bootstrap R^2": bootstrap_results['r2_scores'],
    "Bootstrap MAE": bootstrap_results['mae_scores'],
    "Bootstrap MSE": bootstrap_results['mse_scores']
})
bootstrap_df.to_csv(os.path.join(data_dir, "bootstrap_results.csv"), index=False)

# AIC Calculation (for comparison with cross-validation and bootstrapping)
# Fit the model to the entire dataset and calculate AIC
model.fit(X, y)
y_pred = model.predict(X)

# Calculate residual sum of squares (RSS)
rss = np.sum((y - y_pred) ** 2)

# Number of parameters in the model (including the intercept)
n_params = X.shape[1] + 1  # Number of features + intercept term

# AIC Formula: AIC = n * log(RSS/n) + 2k
aic = n_samples * np.log(rss / n_samples) + 2 * n_params

print("\nAIC for the Linear Regression Model:", aic)

# Visualization 1: MSE Distribution for Cross-Validation
plt.figure(figsize=(10, 6))
sns.histplot(k_fold_results['mse_scores'], kde=True, color='blue', label='K-Fold MSE', bins=20)
plt.title('Distribution of MSE for K-Fold Cross-Validation')
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(os.path.join(data_dir, "mse_distribution_k_fold.png"))
plt.show()

# Visualization 2: MSE Distribution for Bootstrapping
plt.figure(figsize=(10, 6))
sns.histplot(bootstrap_results['mse_scores'], kde=True, color='green', label='Bootstrap MSE', bins=20)
plt.title('Distribution of MSE for Bootstrapping')
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(os.path.join(data_dir, "mse_distribution_bootstrap.png"))
plt.show()

# Visualization 3: Learning Curve
# We train the model with increasing portions of the data to show learning curve
train_sizes = np.arange(10, n_samples, step=50)  # Training sizes from 10 to 2000
train_errors = []
test_errors = []

for train_size in train_sizes:
    X_train, y_train = X[:train_size], y[:train_size]
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    train_errors.append(np.mean((y_train - y_train_pred) ** 2))

    y_test_pred = model.predict(X)
    test_errors.append(np.mean((y - y_test_pred) ** 2))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_errors, label='Train Error', color='blue')
plt.plot(train_sizes, test_errors, label='Test Error', color='red')
plt.title('Learning Curve')
plt.xlabel('Number of Training Samples')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.savefig(os.path.join(data_dir, "learning_curve.png"))
plt.show()

# Visualization 4: Residual Plot
# Fit model on entire data
model.fit(X, y)
y_pred = model.predict(X)
residuals = y - y_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.savefig(os.path.join(data_dir, "residual_plot.png"))
plt.show()
