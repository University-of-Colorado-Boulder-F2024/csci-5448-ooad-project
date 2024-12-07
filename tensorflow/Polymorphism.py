# Polymorphism.py
import tensorflow as tf
import numpy as np
class BaseModel(tf.keras.Model):
    def call(self, inputs):
        raise NotImplementedError("Subclasses must implement 'call' method")

class DenseLayerModel(BaseModel):
    def __init__(self, units):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units)

    def call(self, inputs):
        return self.dense(inputs)

# Polymorphic usage
model_a = DenseLayerModel(units=10)
model_b = DenseLayerModel(units=20)

# Using the same pipeline with different models
for model in [model_a, model_b]:
    output = model(tf.random.normal([10, 5]))
    print(output)