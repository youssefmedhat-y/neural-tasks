import numpy as np


class BackpropNetwork:
    def __init__(self, hidden_layers=(8,), eta=0.05, epochs=1000,
                 activation='sigmoid', use_bias=True, random_seed=42):
        self.hidden_layers = tuple(hidden_layers)
        if not self.hidden_layers or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1
                                         for n in self.hidden_layers):
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
        # This form of sigmoid avoids overflow for large negative inputs.
        e = np.exp(-np.abs(net))
        return np.where(net >= 0, 1.0 / (1.0 + e), e / (1.0 + e))

    def _derivative(self, output):
        if self.activation == 'tanh':
            return 1.0 - output**2
        return output * (1.0 - output)

    def _initialize(self, n_features, n_classes):
        rng = np.random.default_rng(self.random_seed)
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
        outputs = [X]
        for weights, bias in zip(self.weights, self.biases):
            net = outputs[-1] @ weights + bias
            outputs.append(self._activate(net))
        return outputs

    def _deltas(self, outputs, target):
        # Output-layer error, followed by hidden-layer errors.
        deltas = [None] * len(self.weights)
        error = target - outputs[-1]
        deltas[-1] = error * self._derivative(outputs[-1])
        for layer in range(len(self.weights) - 2, -1, -1):
            error = self.weights[layer + 1] @ deltas[layer + 1]
            deltas[layer] = error * self._derivative(outputs[layer + 1])
        return deltas

    def fit(self, X, y, progress=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] != 5 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError('Training inputs must be a nonempty finite matrix with five features.')
        if y.shape != (len(X),) or not np.isin(y, [0, 1, 2]).all():
            raise ValueError('Labels must be class IDs 0, 1, or 2, one per sample.')
        targets = np.eye(3)[y.astype(int)]  # Same one-hot targets for both activations.
        rng = self._initialize(5, 3)
        self.loss_history = []
        for epoch in range(1, self.epochs + 1):
            # Shuffle training order without changing the train/test split.
            for index in rng.permutation(len(X)):
                outputs = self._forward(X[index])
                deltas = self._deltas(outputs, targets[index])
                # Update after calculating every layer's error.
                for layer, delta in enumerate(deltas):
                    self.weights[layer] += self.eta * np.outer(outputs[layer], delta)
                    if self.use_bias:
                        self.biases[layer] += self.eta * delta
            prediction = self._forward(X)[-1]
            loss = float(np.mean(0.5 * np.sum((targets - prediction)**2, axis=1)))
            if not np.isfinite(loss) or not all(np.isfinite(w).all() for w in self.weights):
                raise ValueError('Training diverged. Try a smaller learning rate.')
            self.loss_history.append(loss)
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
        return np.argmax(self.predict_scores(X), axis=1)

    def predict_one_hot(self, X):
        return np.eye(3, dtype=int)[self.predict(X)]

    def predict_single(self, x):
        return int(self.predict(np.asarray(x)[None, :])[0])
