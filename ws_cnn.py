import gzip
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.utils import to_categorical
import pickle
import tensorflow as tf


# load MNIST input data from file
def load_mnist_images(file):
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(file, 'rb') as f:
        image_data = np.frombuffer(f.read(), np.uint8, offset=16)

    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    image_data = image_data.reshape(-1, 1, 28, 28)
    # The inputs come as bytes, we convert them to  float32 [0,1] normalized from range [0,256].
    return image_data / np.float32(256)


# load MNIST output labels
def load_mnist_labels(file):
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(file, 'rb') as f:
        label_data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now
    return label_data


# load MNIST train and test dataset
def load_dataset_mnist():
    # Training Data
    X_train = load_mnist_images('MNIST_Dataset/train-images-idx3-ubyte.gz')
    Y_train = load_mnist_labels('MNIST_Dataset/train-labels-idx1-ubyte.gz')

    # Test Data
    X_test = load_mnist_images('MNIST_Dataset/t10k-images-idx3-ubyte.gz')
    Y_test = load_mnist_labels('MNIST_Dataset/t10k-labels-idx1-ubyte.gz')

    return X_train, Y_train, X_test, Y_test


# Print image from dataset
# Input: Image in (28*28) grayscale
def print_image(image_data):
    plt.imshow(image_data)
    plt.show()


# Split Training Dataset based on numbers
# Input : X_train,Y_train - Training Data
#         X_test,Y_test - Test Data
# Output: Split Training and Test Data
def split_train_test_data(X_train, Y_train, X_test, Y_test):
    split_category1 = [0, 1, 2, 3, 4, 9]
    split_category2 = [5, 6, 7, 8]
    Y_train_categorical = to_categorical(Y_train)
    Y_test_categorical = to_categorical(Y_test)

    # Split Training Data
    X_train1 = np.array([X_train[i, 0] for i, x in enumerate(Y_train) if x in split_category1])
    Y_train1 = np.array([Y_train_categorical[i] for i, x in enumerate(Y_train) if x in split_category1])

    X_train2 = np.array([X_train[i, 0] for i, x in enumerate(Y_train) if x in split_category2])
    Y_train2 = np.array([Y_train_categorical[i] for i, x in enumerate(Y_train) if x in split_category2])

    # Split Test Data
    X_test1 = np.array([X_test[i, 0] for i, x in enumerate(Y_test) if x in split_category1])
    Y_test1 = np.array([Y_test_categorical[i] for i, x in enumerate(Y_test) if x in split_category1])

    X_test2 = np.array([X_test[i, 0] for i, x in enumerate(Y_test) if x in split_category2])
    Y_test2 = np.array([Y_test_categorical[i] for i, x in enumerate(Y_test) if x in split_category2])

    return X_train1, Y_train1, X_train2, Y_train2, X_test1, Y_test1, X_test2, Y_test2


# Create Neural Networks from keras package
# Output: model - Initialized ANN with specified architecture
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Perform Training of Models and save models locally for future use
# Inputs : X_train1, Y_train1, X_train2, Y_train2 - Training Data split into subsets
# Outputs: ann_model1,ann_model2 - Trained ANN Models
def train_cnn_models(X_train1, Y_train1, X_train2, Y_train2):
    # Load/Creata ANN for first group of data
    try:
        ann_model1 = pickle.load(open('ann_model1.sav', 'rb'))
    except IOError:
        ann_model1 = create_cnn_model()
        # Add 1 channel field(for Grayscale)
        X_train1 = X_train1.reshape(X_train1.shape[0], 28, 28, 1)

        # Perform Training
        ann_model1.fit(X_train1, Y_train1, epochs=10, batch_size=32, verbose=0)

        # Save Model for future use
        pickle.dump(ann_model1, open('ann_model1.sav', 'wb'))

    # Load/Creata ANN for second group of data
    try:
        ann_model2 = pickle.load(open('ann_model2.sav', 'rb'))
    except IOError:
        ann_model2 = create_cnn_model()
        # Add 1 channel field(for Grayscale)
        X_train2 = X_train2.reshape(X_train2.shape[0], 28, 28, 1)

        # Perform Training
        ann_model2.fit(X_train2, Y_train2, epochs=10, batch_size=32, verbose=0)

        # Save Model for future use
        pickle.dump(ann_model2, open('ann_model2.sav', 'wb'))

    return ann_model1, ann_model2


# Prints accuracy of model
# Inputs: model  - ANN model
#         X_test - Testing Feature Data
#         Y_test - Testing Data Output Label
def get_accuracy(model, X_test, Y_test):
    # Reshape Training data for grayscale channel
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    _, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Accuracy of model> %.3f' % (acc * 100.0))


# Perform knowledge transfer using Weight Summation Methodology
# Inputs:  model1 - ANN trained on classes c1
#          model2 - ANN trained on classes c2
# Output:  ann_output_model - Consolidated ANN using Weight Summation Technique on classes (c1 U c2)
def perform_weight_summation(model1, model2):
    # Get MetaData for Model1
    layer_weights_1 = {}
    layer_biases_1 = {}
    for layer_number in range(len(model1.layers)):
        if layer_number == 1 or layer_number == 2:  # Skip weight collection for Pooling and Flatten Layers
            continue
        layer_weights_1[layer_number] = model1.layers[layer_number].get_weights()[0]
        layer_biases_1[layer_number] = model1.layers[layer_number].get_weights()[1]

    # Get MetaData for Model2
    layer_weights_2 = {}
    layer_biases_2 = {}
    for layer_number in range(len(model2.layers)):
        if layer_number == 1 or layer_number == 2:  # Skip weight collection for Pooling and Flatten Layers
            continue
        layer_weights_2[layer_number] = model2.layers[layer_number].get_weights()[0]
        layer_biases_2[layer_number] = model2.layers[layer_number].get_weights()[1]

    ann_output_model = create_cnn_model()
    for layer_number in range(len(ann_output_model.layers)):
        if layer_number == 1 or layer_number == 2:  # Skip weight aggregation for Pooling and Flatten Layers
            continue
        ann_output_model.layers[layer_number].set_weights([layer_weights_1[layer_number]+ layer_weights_2[layer_number],
                                                           layer_biases_1[layer_number]+layer_biases_2[layer_number]])

    return ann_output_model

# Main Function
def main():
    # Get MNIST dataset
    X_train, Y_train, X_test, Y_test = load_dataset_mnist()

    # Split Training Data and Test data for Different Data Models
    X_train1, Y_train1, X_train2, Y_train2, X_test1, Y_test1, X_test2, Y_test2 = split_train_test_data(X_train, Y_train, X_test, Y_test)

    # Get Trained Models for network1 and network2
    ann_model1, ann_model2 = train_cnn_models(X_train1, Y_train1, X_train2, Y_train2)

    # Build Consolidated ANN using Weight Summation Technique
    ann_output_model = perform_weight_summation(ann_model1, ann_model2)

    # Get Accuracy of Consolidated Model on test data
    get_accuracy(ann_output_model,np.append(X_test1,X_test2,axis=0),np.append(Y_test1,Y_test2,axis=0))


if __name__ == "__main__":
    main()
