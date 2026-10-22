import numpy as np


class Perceptron:
    def __init__(self, eta=0.01, epochs=100, use_bias=True, random_seed=0):
        self.eta = eta
        self.epochs = epochs
        self.use_bias = use_bias
        self.random_seed = random_seed
        self.w1 = 0.0
        self.w2 = 0.0
        self.b = 0.0
        self.epochs_run = 0

    def fit(self, X, y):
        # Initialize weights randomly between -0.01 and 0.01
        rng = np.random.RandomState(self.random_seed)
        self.w1 = float(rng.uniform(-0.01, 0.01))
        self.w2 = float(rng.uniform(-0.01, 0.01))
        self.b = float(rng.uniform(-0.01, 0.01)) if self.use_bias else 0.0

        for epoch in range(1, self.epochs + 1):
            errors = 0
            for i in range(len(X)):
                x1 = X[i][0]
                x2 = X[i][1]
                target = y[i]

                # Net input: v = w1*x1 + w2*x2 + b
                v = self.w1 * x1 + self.w2 * x2 + self.b
                # Signum activation: +1 if v >= 0, else -1
                y_hat = 1 if v >= 0 else -1

                # Error = target - predicted
                error = target - y_hat

                # Update weights if misclassified
                if error != 0:
                    errors += 1
                    self.w1 += self.eta * error * x1
                    self.w2 += self.eta * error * x2
                    if self.use_bias:
                        self.b += self.eta * error

            # Early stopping: stop if all training samples are classified correctly
            if errors == 0:
                self.epochs_run = epoch
                return self

        self.epochs_run = self.epochs
        return self

    def predict_single(self, x):
        v = self.w1 * x[0] + self.w2 * x[1] + self.b
        return 1 if v >= 0 else -1

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_single(x))
        return np.array(predictions)

    def get_decision_boundary(self):
        # Line equation: w1*x1 + w2*x2 + b = 0
        return self.w1, self.w2, self.b

    def get_mse(self):
        return float('nan')


class Adaline:
    def __init__(self, eta=0.01, epochs=100, mse_threshold=0.01, use_bias=True, random_seed=0):
        self.eta = eta
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.use_bias = use_bias
        self.random_seed = random_seed
        self.w1 = 0.0
        self.w2 = 0.0
        self.b = 0.0
        self.epochs_run = 0
        self.final_mse = 0.0

    def fit(self, X, y):
        # Initialize weights randomly between -0.01 and 0.01
        rng = np.random.RandomState(self.random_seed)
        self.w1 = float(rng.uniform(-0.01, 0.01))
        self.w2 = float(rng.uniform(-0.01, 0.01))
        self.b = float(rng.uniform(-0.01, 0.01)) if self.use_bias else 0.0

        for epoch in range(1, self.epochs + 1):
            # Weight update uses linear activation (before signum)
            for i in range(len(X)):
                x1 = X[i][0]
                x2 = X[i][1]
                target = y[i]

                # Linear output: v = w1*x1 + w2*x2 + b
                v = self.w1 * x1 + self.w2 * x2 + self.b
                error = target - v

                # Adaline update rule
                self.w1 += self.eta * error * x1
                self.w2 += self.eta * error * x2
                if self.use_bias:
                    self.b += self.eta * error

            # Calculate Mean Squared Error (MSE) for this epoch
            total_sq_error = 0.0
            for i in range(len(X)):
                v = self.w1 * X[i][0] + self.w2 * X[i][1] + self.b
                total_sq_error += 0.5 * ((y[i] - v) ** 2)
            mse = total_sq_error / len(X)
            self.final_mse = mse

            # Early stopping if MSE drops below or equal to threshold
            if mse <= self.mse_threshold:
                self.epochs_run = epoch
                return self

        self.epochs_run = self.epochs
        return self

    def predict_single(self, x):
        # Apply signum step function only during prediction
        v = self.w1 * x[0] + self.w2 * x[1] + self.b
        return 1 if v >= 0 else -1

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self.predict_single(x))
        return np.array(predictions)

    def get_decision_boundary(self):
        # Line equation: w1*x1 + w2*x2 + b = 0
        return self.w1, self.w2, self.b

    def get_mse(self):
        return self.final_mse
