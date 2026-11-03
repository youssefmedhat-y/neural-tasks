import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from backprop import BackpropNetwork, evaluate
from task2_data import load_task2, Preprocessor
from data_loader import CLASS_NAMES, FEATURE_NAMES

CSV = Path(__file__).with_name('penguins.csv')


class Task2Tests(unittest.TestCase):
    def test_fixed_split_and_no_leakage(self):
        data = load_task2(CSV)
        self.assertEqual(data.X_train.shape, (90, 5))
        self.assertEqual(data.X_test.shape, (60, 5))
        self.assertFalse(set(data.train_frame.index) & set(data.test_frame.index))
        raw = pd.read_csv(CSV)
        for name in CLASS_NAMES:
            indices = raw.index[raw.Species == name].tolist()
            self.assertEqual(data.train_frame.query('Species == @name').index.tolist(), indices[:30])
            self.assertEqual(data.test_frame.query('Species == @name').index.tolist(), indices[30:])
        self.assertTrue(np.isfinite(data.X_train).all())
        np.testing.assert_allclose(data.X_train.mean(axis=0), 0, atol=1e-12)
        # Extreme test changes cannot influence training preprocessing.
        raw.loc[data.test_frame.index, 'BodyMass'] = 999999.0
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / 'altered.csv'
            raw.to_csv(altered, index=False)
            other = load_task2(altered)
        np.testing.assert_array_equal(data.X_train, other.X_train)
        # A single sample uses exactly the same transform as a batch.
        row = data.test_frame.iloc[0][FEATURE_NAMES].tolist()
        np.testing.assert_allclose(data.preprocessor.single(row), data.X_test[0])

    def test_backprop_gradients(self):
        # Finite differences independently check every weight and bias across 3 layers.
        rng = np.random.default_rng(9)
        x, target = rng.normal(size=5), np.array([0., 1., 0.])
        for activation in ('sigmoid', 'tanh'):
            model = BackpropNetwork((3, 2), activation=activation)
            model._initialize(5, 3)
            outputs = model._forward(x)
            deltas = model._deltas(outputs, target)
            loss = lambda: 0.5 * np.sum((target - model._forward(x)[-1])**2)
            for layer in range(len(model.weights)):
                for array, expected in [(model.weights[layer], -np.outer(outputs[layer], deltas[layer])),
                                        (model.biases[layer], -deltas[layer])]:
                    for index in np.ndindex(array.shape):
                        old = array[index]
                        array[index] = old + 1e-6
                        plus = loss()
                        array[index] = old - 1e-6
                        minus = loss()
                        array[index] = old
                        self.assertAlmostEqual((plus - minus) / 2e-6, expected[index], places=7)

    def test_learning_reproducibility_and_no_bias(self):
        data = load_task2(CSV)
        for activation in ('sigmoid', 'tanh'):
            model = BackpropNetwork((8,), activation=activation, epochs=150).fit(data.X_train, data.y_train)
            self.assertLess(model.loss_history[-1], model.loss_history[0])
            self.assertGreater(evaluate(model, data.X_train, data.y_train)['accuracy'], 80)
            one_hot = model.predict_one_hot(data.X_test)
            np.testing.assert_array_equal(one_hot.sum(axis=1), np.ones(60))
            self.assertEqual(np.asarray(evaluate(model, data.X_test, data.y_test)['confusion_matrix']).sum(), 60)
        kwargs = dict(hidden_layers=(4, 3), epochs=4, use_bias=False)
        a = BackpropNetwork(**kwargs).fit(data.X_train, data.y_train)
        b = BackpropNetwork(**kwargs).fit(data.X_train, data.y_train)
        for bias in a.biases:
            self.assertTrue((bias == 0).all())
        for first, second in zip(a.weights, b.weights):
            np.testing.assert_array_equal(first, second)

    def test_invalid_input_and_metrics(self):
        for kwargs in [dict(hidden_layers=()), dict(hidden_layers=(0,)), dict(eta=float('nan')),
                       dict(eta=0), dict(epochs=0), dict(activation='relu')]:
            with self.assertRaises(ValueError):
                BackpropNetwork(**kwargs)
        model = BackpropNetwork()
        with self.assertRaises(RuntimeError):
            model.predict(np.zeros((1, 5)))
        class Fixed:
            def predict(self, X):
                return np.array([0, 2, 2, 1])
        result = evaluate(Fixed(), None, np.array([0, 1, 2, 2]))
        self.assertEqual(result['accuracy'], 50)
        self.assertEqual(result['confusion_matrix'], [[1, 0, 0], [0, 0, 1], [0, 1, 1]])
        data = load_task2(CSV)
        with self.assertRaises(ValueError):
            data.preprocessor.single([40, 18, 190, 'Unknown', 4000])
        with self.assertRaises(ValueError):
            data.preprocessor.single([float('inf'), 18, 190, 'Dream', 4000])


if __name__ == '__main__':
    unittest.main(verbosity=2)
