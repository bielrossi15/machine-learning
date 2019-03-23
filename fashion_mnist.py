from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.90):
            print('\n\nAlready at 90%!\n\n')
            self.model.stop_training=True

# loading dataset
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# normalizing images
train_x = train_x / 255
test_x = test_x / 255

# creating an instance for your callback class
callback = myCallback()

# creating model
model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# compiling model using ADAM optimizer, and setting accuracy as our metric
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs = 10, callbacks=[callback])
model.evaluate(test_x, test_y)

test_example_index = random.randint(0, 100)
test_example = np.array(test_x[test_example_index])
print(model.predict_classes(test_example.reshape(1, 28, 28)))
plt.imshow(test_example)
plt.show()