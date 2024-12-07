import tensorflow_decision_forests as tfdf
import numpy as np
import pandas as pd
import time
import psutil
import os
import matplotlib.pyplot as plt

class PerformanceBenchmark:
    def _init_(self, num_trees=10, max_depth=3):  
        self.model = tfdf.keras.RandomForestModel(num_trees=num_trees, max_depth=max_depth)
        self.performance_metrics = {}

    def train(self, train_ds):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # Memory in MB
        start_time = time.time()

        self.model.fit(train_ds)

        end_time = time.time()
        mem_after = process.memory_info().rss / (1024 * 1024)
        cpu_time = end_time - start_time

        print(f"TensorFlow: Training Time: {cpu_time:.2f}s")
        print(f"TensorFlow: Memory Usage: {mem_after - mem_before:.2f} MB")

        self.performance_metrics['training_time'] = round(cpu_time, 2)
        self.performance_metrics['memory_usage'] = round(mem_after - mem_before, 2)

    def predict(self, X):
        start_time = time.time()
        predictions = self.model.predict(X)
        end_time = time.time()

        prediction_time = end_time - start_time
        print(f"TensorFlow: Prediction Time: {prediction_time:.2f}s")

        self.performance_metrics['prediction_time'] = round(prediction_time, 2)
        return predictions

    def plot_performance(self):
        metrics = ['Training Time (s)', 'Memory Usage (MB)', 'Prediction Time (s)']
        values = [self.performance_metrics.get('training_time', 0),
                  self.performance_metrics.get('memory_usage', 0),
                  self.performance_metrics.get('prediction_time', 0)]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(metrics, values, color=['blue', 'orange', 'green'])
        plt.title("Performance Metrics for TensorFlow RandomForest")
        plt.ylabel("Value")
        plt.xlabel("Metrics")

        # Add numerical values on top of the bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{value}', ha='center', va='bottom')

        plt.show()


# Generate mock data for TensorFlow
X_train = np.random.rand(1000, 10)
y_train = np.random.randint(2, size=1000)
train_ds = tfdf.keras.utils.pd_dataframe_to_tf_dataset(
    pd.DataFrame({'feature': list(X_train), 'label': y_train}),
    label="label"
)

benchmark_tf = PerformanceBenchmark(num_trees=50, max_depth=5)
benchmark_tf.train(train_ds)

X_test = np.random.rand(100, 10)
benchmark_tf.predict(X_test)

# Plot the performance metrics
benchmark_tf.plot_performance()