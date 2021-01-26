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
from keras.models import load_model


fisher_information = {}
is_first = False

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
            #current_loss = tf.reduce_mean(current_loss)
            dE_dW = tape_2.gradient(current_loss, model.trainable_weights)
        d2E_dW2 = tape_1.gradient(dE_dW,model.trainable_weights)

    global fisher_information
    global is_first
    if is_first == False:
        fisher_information = d2E_dW2
        is_first = True

    for layer_nodes in range(6):   #Hardcoded for time being
        fisher_information[layer_nodes] = tf.add(fisher_information[layer_nodes],d2E_dW2[layer_nodes])


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
# Outputs: Locally Save Fisher Information Numpy variables for each model
def training_models_fisher_computation(X_train1, Y_train1, X_train2, Y_train2, X_test1, Y_test1, X_test2, Y_test2):

    global fisher_information
    global is_first
    #################################################
    ####### Model 1 Training Parameters and Training
    #################################################
    ann_model1 = create_cnn_model()

    # Add 1 channel field(for Grayscale)
    X_train1 = X_train1.reshape(X_train1.shape[0], 28, 28, 1)

    # Perform Training to get Fisher Information
    fisher_callback = CustomCallBack()
    ann_model1.fit(X_train1, Y_train1, epochs=10, batch_size=32, verbose=1, callbacks=[fisher_callback])

    # Save Model for future use
    ann_model1.save('ann_ewc_model1.h5')

    # Save numpy variables in File
    layer_counter = 0

    for layer_nodes in fisher_information:
        np.save('Model1FI/'+str(layer_counter),layer_nodes.numpy())
        layer_counter = layer_counter+1

    ###############################################
    ##### Model 2 Training Parameters and Training
    ###############################################
    fisher_information = {}
    is_first = False
    ann_model2 = create_cnn_model()

    # Add 1 channel field(for Grayscale)
    X_train2 = X_train2.reshape(X_train2.shape[0], 28, 28, 1)

    # Perform Training to get Fisher Information
    fisher_callback = CustomCallBack()
    ann_model2.fit(X_train2, Y_train2, epochs=10, batch_size=32, verbose=1, callbacks=[fisher_callback])

    # Save Model for future use
    ann_model2.save('ann_ewc_model2.h5')

    # Save numpy variables in File
    layer_counter = 0
    for layer_nodes in fisher_information:
        np.save('Model2FI/'+str(layer_counter),layer_nodes.numpy())
        layer_counter = layer_counter+1

# Retrieve Fisher Information of trained subset models
# Inputs : None
# Outputs: Get Locally Saved Fisher Information Numpy variables for each model
def get_fisher_information():
    # Model 1 Parameters
    layer_weights_1 = {}
    layer_biases_1 = {}
    for loadfile_counter in range(6):
        if loadfile_counter % 2 == 0:
            layer_weights_1[(int)(loadfile_counter/2)] = np.load('Model1FI/'+str(loadfile_counter)+'.npy')
        else:
            layer_biases_1[(int)(loadfile_counter/2)] = np.load('Model1FI/'+str(loadfile_counter)+'.npy')

    # Model 2 Parameters
    layer_weights_2 = {}
    layer_biases_2 = {}
    for loadfile_counter in range(6):
        if loadfile_counter % 2 == 0:
            layer_weights_2[(int)(loadfile_counter/2)] = np.load('Model2FI/'+str(loadfile_counter)+'.npy')
        else:
            layer_biases_2[(int)(loadfile_counter/2)] = np.load('Model2FI/'+str(loadfile_counter)+'.npy')

    return layer_weights_1, layer_biases_1, layer_weights_2, layer_biases_2

# Retrieve Weights and Biases of trained subset models
# Inputs : None
# Outputs: Get Locally Saved Numpy variables for each model
def get_saved_model_weights():
    # Model 1 Parameters
    layer_weights_1 = {}
    layer_biases_1 = {}

    ann_model1 = load_model('ann_ewc_model1.h5', custom_objects={'SequentialCustom': SequentialCustom})
    layer_counter = 0
    for layer_number in range(len(ann_model1.layers)):
        if layer_number == 1 or layer_number == 2:
            continue
        layer_weights_1[layer_counter] = ann_model1.layers[layer_number].get_weights()[0]
        layer_biases_1[layer_counter] = ann_model1.layers[layer_number].get_weights()[1]
        layer_counter = layer_counter + 1

    # Model 2 Parameters
    layer_weights_2 = {}
    layer_biases_2 = {}

    ann_model2 = load_model('ann_ewc_model2.h5', custom_objects={'SequentialCustom': SequentialCustom})
    layer_counter = 0
    for layer_number in range(len(ann_model2.layers)):
        if layer_number == 1 or layer_number == 2:
            continue
        layer_weights_2[layer_counter] = ann_model2.layers[layer_number].get_weights()[0]
        layer_biases_2[layer_counter] = ann_model2.layers[layer_number].get_weights()[1]
        layer_counter = layer_counter + 1

    return layer_weights_1, layer_biases_1, layer_weights_2, layer_biases_2

# Perform EWC and get resultant ANN
# Inputs : None
# Outputs: Resultant ANN using EWC
def perform_symmetric_ewc():
    # Get Fisher Information
    fisher_layer_weights_1, fisher_layer_biases_1, fisher_layer_weights_2, fisher_layer_biases_2 = get_fisher_information()

    # Get model weights and parameters
    layer_weights_1, layer_biases_1, layer_weights_2, layer_biases_2 = get_saved_model_weights()

    # Get resultant weights using EWC technique  W(final) = (W(a)*F(a) + W(b)*F(b))/(F(a) + F(b))
    layer_resultant_weights = {}
    layer_resultant_biases = {}

    for layer_counter in range(len(layer_weights_1)):
        norm = np.linalg.norm(fisher_layer_weights_1[layer_counter])
        fisher_layer_weights_1[layer_counter] = fisher_layer_weights_1[layer_counter]/norm

        norm = np.linalg.norm(fisher_layer_biases_1[layer_counter])
        fisher_layer_biases_1[layer_counter] = fisher_layer_biases_1[layer_counter]/norm

        norm = np.linalg.norm(fisher_layer_weights_2[layer_counter])
        fisher_layer_weights_2[layer_counter] = fisher_layer_weights_2[layer_counter]/norm

        norm = np.linalg.norm(fisher_layer_biases_2[layer_counter])
        fisher_layer_biases_2[layer_counter] = fisher_layer_biases_2[layer_counter]/norm

        layer_resultant_weights[layer_counter] = np.divide(np.add(np.multiply(layer_weights_1[layer_counter],fisher_layer_weights_1[layer_counter]),
        np.multiply(layer_weights_2[layer_counter],fisher_layer_weights_2[layer_counter])), np.add(fisher_layer_weights_1[layer_counter],fisher_layer_weights_2[layer_counter]))

        layer_resultant_biases[layer_counter] = np.divide(np.add(np.multiply(layer_biases_1[layer_counter],fisher_layer_biases_1[layer_counter]),
        np.multiply(layer_biases_2[layer_counter],fisher_layer_biases_2[layer_counter])), np.add(fisher_layer_biases_1[layer_counter],
                                                                                                 fisher_layer_biases_2[layer_counter]))

    # Assign weights to new model
    ann_output_model = create_cnn_model()
    layer_counter = 0
    for layer_number in range(len(ann_output_model.layers)):
        if layer_number == 1 or layer_number == 2:  # Skip weight assignment for Pooling and Flatten Layers
            continue
        ann_output_model.layers[layer_number].set_weights([layer_resultant_weights[layer_counter],
                                                           layer_resultant_biases[layer_counter]])
        layer_counter = layer_counter + 1

    return ann_output_model


def main():

    # Get MNIST dataset
    X_train, Y_train, X_test, Y_test = ws.load_dataset_mnist()

    # Split Training Data and Test data for Different Data Models
    X_train1, Y_train1, X_train2, Y_train2, X_test1, Y_test1, X_test2, Y_test2 = ws.split_train_test_data(X_train, Y_train, X_test, Y_test)

    # Perform Training and saving of Fisher Information
    training_models_fisher_computation(X_train1, Y_train1, X_train2, Y_train2, X_test1, Y_test1, X_test2, Y_test2)

    # Perform Elastic Weight Consolidation for Result ANN
    ann_output_model = perform_symmetric_ewc()


    # Get Accuracy of Consolidated Model on test data
    ws.get_accuracy(ann_output_model,X_test1,Y_test1)
    ws.get_accuracy(ann_output_model,X_test2,Y_test2)
    ws.get_accuracy(ann_output_model,np.append(X_test1,X_test2,axis=0),np.append(Y_test1,Y_test2,axis=0))


if __name__ == "__main__":
    main()
