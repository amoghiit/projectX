import numpy as np

class ModelSelection:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def mean_absolute_error(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r_squared(self, y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2) 
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def k_fold_cross_validation(self, k=5):
        fold_size = len(self.X) // k
        r2_scores = []
        mae_scores = []
        mse_scores = []

        for i in range(k):
            # Split data into k folds
            start = i * fold_size
            end = (i + 1) * fold_size if i != k - 1 else len(self.X)

            X_train = np.concatenate([self.X[:start], self.X[end:]])
            y_train = np.concatenate([self.y[:start], self.y[end:]])
            X_val = self.X[start:end]
            y_val = self.y[start:end]

            # Train model
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)

            # Evaluate performance
            r2 = self.r_squared(y_val, y_pred)
            mae = self.mean_absolute_error(y_val, y_pred)
            mse = self.mean_squared_error(y_val, y_pred)

            r2_scores.append(r2)
            mae_scores.append(mae)
            mse_scores.append(mse)

        return {
            'mean_r2': np.mean(r2_scores),
            'mean_mae': np.mean(mae_scores),
            'mean_mse': np.mean(mse_scores),
            'r2_scores': r2_scores,
            'mae_scores': mae_scores,
            'mse_scores': mse_scores
        }

    def bootstrap(self, n_iterations=100):
        r2_scores = []
        mae_scores = []
        mse_scores = []

        for i in range(n_iterations):
            # Resample data with replacement
            indices = np.random.choice(len(self.X), size=len(self.X), replace=True)
            X_resampled = self.X[indices]
            y_resampled = self.y[indices]

            # Train model
            self.model.fit(X_resampled, y_resampled)
            y_pred = self.model.predict(self.X)

            # Evaluate performance
            r2 = self.r_squared(self.y, y_pred)
            mae = self.mean_absolute_error(self.y, y_pred)
            mse = self.mean_squared_error(self.y, y_pred)

            r2_scores.append(r2)
            mae_scores.append(mae)
            mse_scores.append(mse)

        return {
            'mean_r2': np.mean(r2_scores),
            'mean_mae': np.mean(mae_scores),
            'mean_mse': np.mean(mse_scores),
            'r2_scores': r2_scores,
            'mae_scores': mae_scores,
            'mse_scores': mse_scores
        }
