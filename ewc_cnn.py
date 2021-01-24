import tensorflow as tf
import numpy as np
import ws_cnn as ws
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.python.eager import backprop


fisher_information = {}

# Override tf.keras.Model class to store x_train and y_train during each batch processing
# Used later in Callback method
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

        # Send value for Hessian/ Fisher Information Calculation
        global x_batch, y_batch
        x_batch = x
        y_batch = y

        return {m.name: m.result() for m in self.metrics}

# Update Fisher information during Callback
# Inputs : model, x_batch, y_batch
# Outputs: Global Variable fisher_information updated
def get_fisherinformation(model,x_batch,y_batch):
    with tf.GradientTape() as tape_1:
        with tf.GradientTape() as tape_2:
            preds = model(x_batch)
            current_loss = tf.keras.losses.categorical_crossentropy(y_batch,preds,from_logits=True)
            current_loss = tf.reduce_mean(current_loss)
            dE_dW = tape_2.gradient(current_loss, model.trainable_weights)
        d2E_dW2 = tape_1.gradient(dE_dW,model.trainable_weights)

    global fisher_information
    fisher_information = d2E_dW2

# Compute Hessians from model
# Inputs : model, x_batch, y_batch , y_predicted
# Outputs: Hessian Tensor Variable
def get_hessian(model,x_batch,y_batch,y_predicted):
    with tf.GradientTape(persistent=True) as tape:
        preds = model(x_batch)
        loss = tf.keras.losses.categorical_crossentropy(y_batch,preds,from_logits=True)
        loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        flattened_grads = tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)
    tf.print(tf.shape(flattened_grads))
    hessians = [tape.jacobian(grad, model.trainable_variables) for grad in grads]
    flattened_hessians = tf.concat([tf.reshape(hess, [hess.shape[0], -1]) for hess in hessians], 1)
    return flattened_hessians

# Create Custom CallBack for triggering Fisher Information Calculation
class CustomCallBack(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        #flatenned_hessians = get_hessian(self.model,x_batch,y_batch,y_predicted)
        get_fisherinformation(self.model,x_batch,y_batch)

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

# Call Builder Method for Fisher Information Computation of Subset Models
# Inputs : None
# Outputs: Locally Saved Fisher Information Numpy variables for each model
def training_models_fisher_computation():

    global fisher_information
    # Get MNIST dataset
    X_train, Y_train, X_test, Y_test = ws.load_dataset_mnist()

    # Split Training Data and Test data for Different Data Models
    X_train1, Y_train1, X_train2, Y_train2, X_test1, Y_test1, X_test2, Y_test2 = ws.split_train_test_data(X_train, Y_train, X_test, Y_test)

    #################################################
    ####### Model 1 Training Parameters and Training
    #################################################
    ann_model1 = create_cnn_model()

    # Add 1 channel field(for Grayscale)
    X_train1 = X_train1.reshape(X_train1.shape[0], 28, 28, 1)

    # Perform Training to get Fisher Information
    fisher_callback = CustomCallBack()
    ann_model1.fit(X_train1, Y_train1, epochs=1, batch_size=32, verbose=1, callbacks=[fisher_callback])

    # Save numpy variables in File
    layer_counter = 0
    for layer_nodes in fisher_information:
        np.save('Model1FI/'+str(layer_counter),layer_nodes.numpy())
        layer_counter = layer_counter+1

    ###############################################
    ##### Model 2 Training Parameters and Training
    ###############################################
    fisher_information = {}
    ann_model2 = create_cnn_model()

    # Add 1 channel field(for Grayscale)
    X_train2 = X_train2.reshape(X_train2.shape[0], 28, 28, 1)

    # Perform Training to get Fisher Information
    fisher_callback = CustomCallBack()
    ann_model1.fit(X_train2, Y_train2, epochs=1, batch_size=32, verbose=1, callbacks=[fisher_callback])

    # Save numpy variables in File
    layer_counter = 0
    for layer_nodes in fisher_information:
        np.save('Model2FI/'+str(layer_counter),layer_nodes.numpy())
        layer_counter = layer_counter+1

def main():
    training_models_fisher_computation()


if __name__ == "__main__":
    main()
