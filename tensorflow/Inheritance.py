#Inheritance
import tensorflow as tf

# Define a base class for models
class BaseModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__(**kwargs)

    def call(self, inputs):
        raise NotImplementedError("Subclasses must implement the call method.")

# Define a derived class for a specific model
class CustomModel(BaseModel):
    def __init__(self, num_units):
        super(CustomModel, self).__init__()
        self.dense_layer = tf.keras.layers.Dense(num_units, activation='relu')

    def call(self, inputs):
        return self.dense_layer(inputs)

# Instantiate and use the custom model
model = CustomModel(num_units=32)
output = model(tf.random.uniform([1, 10]))  # Passing random input
print(output)