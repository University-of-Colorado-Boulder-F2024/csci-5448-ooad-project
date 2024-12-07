import tensorflow_decision_forests as tfdf
import numpy as np

class RandomForestModel:
    def __init__(self, num_trees=10, max_depth=5):
        # Encapsulating model parameters
        self.num_trees = num_trees
        self.max_depth = max_depth
        # Encapsulating the TensorFlow model
        self.model = tfdf.keras.RandomForestModel(
            num_trees=self.num_trees, max_depth=self.max_depth
        )

    def train(self, train_ds):
        """Train the model on the given dataset."""
        self.model.fit(train_ds)

    def predict(self, X):
        """Predict the labels for the given data."""
        predictions = self.model.predict(X)
        return np.round(predictions).astype(int)