import unittest
import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf
from performance_analysis import PerformanceBenchmark

class TestPerformanceBenchmark(unittest.TestCase):
    def setUp(self):
        """Set up data and benchmarking instance for tests."""
        self.benchmark = PerformanceBenchmark(num_trees=10, max_depth=3)
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(2, size=100)
        self.train_ds = tfdf.keras.utils.pd_dataframe_to_tf_dataset(
            pd.DataFrame({'feature': list(self.X_train), 'label': self.y_train}),
            label="label"
        )
        self.X_test = np.random.rand(10, 10)

    def test_initialization(self):
        """Test if the PerformanceBenchmark instance initializes correctly."""
        self.assertIsNotNone(self.benchmark)
        self.assertEqual(self.benchmark.model.num_trees, 10)
        self.assertEqual(self.benchmark.model.max_depth, 3)

    def test_training(self):
        """Test the training process."""
        try:
            self.benchmark.train(self.train_ds)
            self.assertIn('training_time', self.benchmark.performance_metrics)
            self.assertIn('memory_usage', self.benchmark.performance_metrics)
        except Exception as e:
            self.fail(f"Training failed with exception: {e}")

    def test_prediction(self):
        """Test the prediction process."""
        self.benchmark.train(self.train_ds)
        predictions = self.benchmark.predict(self.X_test)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), 10)

    def test_plot_performance(self):
        """Test if the performance plot can be generated without errors."""
        self.benchmark.train(self.train_ds)
        self.benchmark.predict(self.X_test)
        try:
            self.benchmark.plot_performance()
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")

if _name_ == '_main_':
    unittest.main()