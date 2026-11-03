import numpy as np


class BackpropNetwork:
    def __init__(self, hidden_layers=(8,), eta=0.05, epochs=1000,
                 activation='sigmoid', use_bias=True, random_seed=42):
        self.hidden_layers = tuple(hidden_layers)
        if len(self.hidden_layers) == 0:
            raise ValueError('Use at least one hidden layer with positive integer neuron counts.')
        for neurons in self.hidden_layers:
            if isinstance(neurons, bool) or not isinstance(neurons, (int, np.integer)) or neurons < 1:
                raise ValueError('Use at least one hidden layer with positive integer neuron counts.')
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError('Learning rate must be finite and greater than zero.')
        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError('Epochs must be a positive integer.')
        if activation not in ('sigmoid', 'tanh'):
            raise ValueError('Activation must be sigmoid or tanh.')
        if not isinstance(random_seed, (int, np.integer)) or not 0 <= random_seed < 2**32:
            raise ValueError('Seed must be an integer between 0 and 4294967295.')
        self.eta = float(eta)
        self.epochs = int(epochs)
        self.activation = activation
        self.use_bias = use_bias
        self.random_seed = random_seed
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.epochs_run = 0

    def _activate(self, net):
        if self.activation == 'tanh':
            return np.tanh(net)

        # Sigmoid = 1 / (1 + exp(-net)).
        # Use two equivalent forms to avoid overflow for large negative net.
        e = np.exp(-np.abs(net))
        positive_output = 1.0 / (1.0 + e)
        negative_output = e / (1.0 + e)
        return np.where(net >= 0, positive_output, negative_output)

    def _derivative(self, output):
        if self.activation == 'tanh':
            return 1.0 - output**2
        return output * (1.0 - output)

    def _initialize(self, n_features, n_classes):
        rng = np.random.default_rng(self.random_seed)
        # For example: 5 inputs -> 8 hidden -> 6 hidden -> 3 outputs.
        sizes = (n_features,) + self.hidden_layers + (n_classes,)
        self.weights = []
        self.biases = []
        # Initialize all weights first to keep seeded runs reproducible.
        for layer in range(len(sizes) - 1):
            shape = (sizes[layer], sizes[layer + 1])
            self.weights.append(rng.uniform(-0.1, 0.1, shape))
        for size in sizes[1:]:
            if self.use_bias:
                self.biases.append(rng.uniform(-0.1, 0.1, size))
            else:
                self.biases.append(np.zeros(size))
        return rng

    def _forward(self, X):
        # Keep each layer's output because backpropagation needs it later.
        outputs = [X]
        for layer in range(len(self.weights)):
            inputs = outputs[layer]
            weights = self.weights[layer]
            bias = self.biases[layer]
            net = inputs @ weights + bias
            output = self._activate(net)
            outputs.append(output)
        return outputs

    def _deltas(self, outputs, target):
        deltas = [None] * len(self.weights)

        # Output layer: delta = (desired - actual) * activation derivative.
        actual = outputs[-1]
        error = target - actual
        deltas[-1] = error * self._derivative(actual)

        # Work backwards through the hidden layers.
        for layer in range(len(self.weights) - 2, -1, -1):
            next_weights = self.weights[layer + 1]
            next_delta = deltas[layer + 1]
            actual = outputs[layer + 1]
            error = next_weights @ next_delta
            deltas[layer] = error * self._derivative(actual)
        return deltas

    def fit(self, X, y, progress=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] != 5 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError('Training inputs must be a nonempty finite matrix with five features.')
        if y.shape != (len(X),) or not np.isin(y, [0, 1, 2]).all():
            raise ValueError('Labels must be class IDs 0, 1, or 2, one per sample.')
        # Desired outputs: Adelie [1,0,0], Chinstrap [0,1,0], Gentoo [0,0,1].
        targets = np.eye(3)[y.astype(int)]
        rng = self._initialize(5, 3)
        self.loss_history = []

        for epoch in range(1, self.epochs + 1):
            sample_order = rng.permutation(len(X))
            for index in sample_order:
                sample = X[index]
                target = targets[index]

                # 1. Forward pass.
                outputs = self._forward(sample)

                # 2. Backward pass. Calculate every delta before updating.
                deltas = self._deltas(outputs, target)

                # 3. Update weights and, if enabled, biases.
                for layer in range(len(self.weights)):
                    inputs = outputs[layer]
                    delta = deltas[layer]
                    weight_change = self.eta * np.outer(inputs, delta)
                    self.weights[layer] += weight_change
                    if self.use_bias:
                        self.biases[layer] += self.eta * delta

            # Record mean half-squared error after this epoch.
            prediction = self._forward(X)[-1]
            error = targets - prediction
            squared_error = error**2
            sample_loss = 0.5 * np.sum(squared_error, axis=1)
            loss = float(np.mean(sample_loss))
            if not np.isfinite(loss):
                raise ValueError('Training diverged. Try a smaller learning rate.')
            for weights in self.weights:
                if not np.isfinite(weights).all():
                    raise ValueError('Training diverged. Try a smaller learning rate.')
            self.loss_history.append(loss)

            # Let the GUI display progress every 25 epochs.
            if progress and (epoch == 1 or epoch % 25 == 0 or epoch == self.epochs):
                progress(epoch, loss)
        self.epochs_run = self.epochs
        return self

    def predict_scores(self, X):
        if not self.weights:
            raise RuntimeError('Train the network before classifying samples.')
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 5 or not np.isfinite(X).all():
            raise ValueError('Prediction inputs must be a finite matrix with five features.')
        return self._forward(X)[-1]

    def predict(self, X):
        scores = self.predict_scores(X)
        predictions = np.argmax(scores, axis=1)
        return predictions

    def predict_one_hot(self, X):
        predictions = self.predict(X)
        one_hot = np.eye(3, dtype=int)[predictions]
        return one_hot

    def predict_single(self, x):
        # Add a row dimension so one sample uses the same prediction code.
        sample = np.asarray(x)[None, :]
        prediction = self.predict(sample)
        return int(prediction[0])
