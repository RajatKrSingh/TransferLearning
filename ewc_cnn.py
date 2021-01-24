import tensorflow as tf
import numpy as np
import ws_cnn as ws
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.python.eager import backprop

previous_layer_weights = {}
previous_layer_biases = {}

previous_layer_weights_gradient = {}
previous_layer_biases_gradient = {}

fisher_information_weights = {}
fisher_information_biases = {}

tf.keras.Model
class SequentialCustom(Sequential):
    def train_step(self, data):
        # We can change the default convention for parameters (tuple x, y and weights)
        # and use any data we want.
        x, y = data

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred)

        # Send value for Hessian Calculation
        global y_predicted, x_batch, y_batch
        y_predicted = y_pred
        x_batch = x
        y_batch = y

        return {m.name: m.result() for m in self.metrics}

def get_fisherinformation(model,x_batch,y_batch,y_predicted):
    loss = tf.keras.losses.categorical_crossentropy(y_batch,y_predicted)
    #for layer_num in range(len(model.layers)):
    #    print(model.layers[layer_num].get_weights()[0].shape)




def get_hessian(model,x_batch,y_batch,y_predicted):
    with tf.GradientTape(persistent=True) as tape:
        preds = model(x_batch)
        loss = tf.keras.losses.categorical_crossentropy(y_batch,preds,from_logits=True)
        loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        flattened_grads = tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)
    tf.print(tf.shape(flattened_grads))
    hessians = [tape.jacobian(grad, model.trainable_variables) for grad in grads]
    #hessians = tape.jacobian(flattened_grads, model.trainable_weights)
    flattened_hessians = tf.concat([tf.reshape(hess, [hess.shape[0], -1]) for hess in hessians], 1)
    return flattened_hessians


class CustomCallBack(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        #flatenned_hessians = get_hessian(self.model,x_batch,y_batch,y_predicted)
        get_fisherinformation(self.model,x_batch,y_batch,y_predicted)

# Create Neural Networks from keras package
# Output: model - Initialized ANN with specified architecture
def create_cnn_model():
    model = SequentialCustom()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'],run_eagerly=True)
    return model


def main():

    # Get MNIST dataset
    X_train, Y_train, X_test, Y_test = ws.load_dataset_mnist()

    # Split Training Data and Test data for Different Data Models
    X_train1, Y_train1, X_train2, Y_train2, X_test1, Y_test1, X_test2, Y_test2 = ws.split_train_test_data(X_train, Y_train, X_test, Y_test)

    # Load Configurations from pretrained model of Weight Summation
    sample_model = load_model('ann_model1.h5')

    global previous_layer_weights
    global previous_layer_biases

    ann_model1 = create_cnn_model()

    # Store initial weights in previous weights global variable. Accessed and Updated in Callback
    for layer_num in range(len(ann_model1.layers)):
        if layer_num == 1 or layer_num == 2:
            continue
        # Initialize current weights in global memory
        previous_layer_weights[layer_num] = ann_model1.layers[layer_num].get_weights()[0]
        previous_layer_biases[layer_num] = ann_model1.layers[layer_num].get_weights()[1]

        # Initialize Jacobian gradients
        previous_layer_weights_gradient[layer_num] = np.zeros(ann_model1.layers[layer_num].get_weights()[0].shape)
        previous_layer_biases_gradient[layer_num] = np.zeros(ann_model1.layers[layer_num].get_weights()[1].shape)

        # Initialize Fisher Information
        fisher_information_weights[layer_num] = np.zeros(ann_model1.layers[layer_num].get_weights()[0].shape)
        fisher_information_biases[layer_num] = np.zeros(ann_model1.layers[layer_num].get_weights()[1].shape)


    # Add 1 channel field(for Grayscale)
    X_train1 = X_train1.reshape(X_train1.shape[0], 28, 28, 1)


    # Perform Training
    fisher_callback = CustomCallBack()
    ann_model1.fit(X_train1, Y_train1, epochs=1, batch_size=32, verbose=1, callbacks=[fisher_callback])


if __name__ == "__main__":
    main()
