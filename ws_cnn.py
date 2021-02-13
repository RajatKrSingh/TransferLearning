import gzip
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
import pickle
import h5py
import collections

# Load USPS dataset
def load_USPS():
    with h5py.File('usps.h5', 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

    X_tr = X_tr.reshape((X_tr.shape[0],16,16))
    X_te = X_te.reshape((X_te.shape[0],16,16))
    return X_tr,y_tr,X_te,y_te

# load CIFAR-10 Dataset
def load_cifar10(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    X = dict['data']
    Y = dict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)
    return X, Y

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

# Split Training Dataset based on numbers CIFAR-10
# Input : X_train,Y_train - Training Data
#         X_test,Y_test - Test Data
# Output: Split Training and Test Data
def split_train_test_data_c10(X_train, Y_train, X_test, Y_test):
    split_category1 = [0, 1, 2, 3, 4, 9]
    split_category2 = [5, 6, 7, 8]
    Y_train_categorical = to_categorical(Y_train)
    Y_test_categorical = to_categorical(Y_test)

    # Split Training Data
    X_train1 = np.array([X_train[i] for i, x in enumerate(Y_train) if x in split_category1])
    Y_train1 = np.array([Y_train_categorical[i] for i, x in enumerate(Y_train) if x in split_category1])

    X_train2 = np.array([X_train[i] for i, x in enumerate(Y_train) if x in split_category2])
    Y_train2 = np.array([Y_train_categorical[i] for i, x in enumerate(Y_train) if x in split_category2])

    # Split Test Data
    X_test1 = np.array([X_test[i] for i, x in enumerate(Y_test) if x in split_category1])
    Y_test1 = np.array([Y_test_categorical[i] for i, x in enumerate(Y_test) if x in split_category1])

    X_test2 = np.array([X_test[i] for i, x in enumerate(Y_test) if x in split_category2])
    Y_test2 = np.array([Y_test_categorical[i] for i, x in enumerate(Y_test) if x in split_category2])

    return X_train1, Y_train1, X_train2, Y_train2, X_test1, Y_test1, X_test2, Y_test2


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
def create_cnn_model_c10():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create Neural Networks from keras package
# Output: model - Initialized ANN with specified architecture
def create_cnn_model_usps():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(16, 16, 1))) #Change to (28,28,1) for MNIST
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create Neural Networks from keras package
# Output: model - Initialized ANN with specified architecture
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1))) #Change to (28,28,1) for MNIST
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_complete_model(X_train,Y_train):
    try:
        ann_model = load_model('ann_model_c10_compete.h5')
    except IOError:
        ann_model = create_cnn_model_c10()
        ann_model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=1)

        ann_model.save('ann_model_c10_compete.h5')

    return ann_model

def transfer_base_learned_features(ann_model1,ann_base_model):
    for layer_number in range(len(ann_base_model.layers)):
        if  layer_number == 11: # Skip weight transfer and freezing
            continue
        else:
            ann_model1.layers[layer_number] = ann_base_model.layers[layer_number]
            ann_model1.layers[layer_number].trainable = False

            if layer_number == 0 or layer_number == 1 or layer_number == 4 or layer_number == 6 or layer_number == 10:
                ann_model1.layers[layer_number].set_weights(ann_base_model.layers[layer_number].get_weights())


    opt = SGD(lr=0.01, momentum=0.9)
    ann_model1.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return ann_model1

# Perform Training of Models and save models locally for future use
# Inputs : X_train1, Y_train1, X_train2, Y_train2 - Training Data split into subsets
# Outputs: ann_model1,ann_model2 - Trained ANN Models
# Model Naming Nomenclature MNIST- ann_model1.h5 CIFAR-10 ann_model1_c10.h5
def train_cnn_models(X_train1, Y_train1, X_train2, Y_train2,ann_base_model):
    # Load/Creata ANN for first group of data
    try:
        ann_model1 = load_model('ann_model1_c10.h5')
    except IOError:
        ann_model1 = create_cnn_model_c10()
        transfer_base_learned_features(ann_model1,ann_base_model)
        # Add 1 channel field(for Grayscale)
        #X_train1 = X_train1.reshape(X_train1.shape[0], 16, 16, 1)

        # Perform Training
        ann_model1.fit(X_train1, Y_train1, epochs=50, batch_size=32, verbose=1)

        # Save Model for future use
        ann_model1.save('ann_model1_c10.h5')

    # Load/Creata ANN for second group of data
    try:
        ann_model2 = load_model('ann_model2_c10.h5')
    except IOError:
        ann_model2 = create_cnn_model_c10()

        # Add 1 channel field(for Grayscale)
        #X_train2 = X_train2.reshape(X_train2.shape[0], 16, 16, 1)
        transfer_base_learned_features(ann_model2,ann_base_model)

        # Perform Training
        ann_model2.fit(X_train2, Y_train2, epochs=50, batch_size=32, verbose=1)

        # Save Model for future use
        ann_model2.save('ann_model2_c10.h5')

    return ann_model1, ann_model2

def get_accuracy_labelwise(model, X_test, Y_test):
    for y_label in range(10):
        X_test_label = np.array([X_test[i] for i, x in enumerate(Y_test) if x[y_label] == 1])
        Y_test_label = np.array([Y_test[i] for i, x in enumerate(Y_test) if x[y_label] == 1])
        print('Label:',y_label)
        get_accuracy(model,X_test_label,Y_test_label)

def consolidated_cnn_c10():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prints accuracy of model
# Inputs: model  - ANN model
#         X_test - Testing Feature Data
#         Y_test - Testing Data Output Label
def get_accuracy(model, X_test, Y_test):
    # Reshape Training data for grayscale channel
    #X_test = X_test.reshape(X_test.shape[0], 16, 16, 1)
    for labels in range(10):
        X_test_label = np.array([X_test[i] for i, x in enumerate(Y_test) if x[labels] == 1])
        Y_test_label = np.array([Y_test[i] for i, x in enumerate(Y_test) if x[labels] == 1])
        _, acc = model.evaluate(X_test_label, Y_test_label, verbose=0)
        predicted_label = model.predict(X_test_label)
        predicted_label = np.argmax(predicted_label,axis=1)
        print('Labels:',labels,'     Accuracy:',(acc*100.0))
        print(collections.Counter(predicted_label))


    _, acc = model.evaluate(X_test, Y_test, verbose=0)

    print('Accuracy of model> %.3f' % (acc * 100.0))

# Perform knowledge transfer using Weight Summation Methodology
# Inputs:  model1 - ANN trained on classes c1
#          model2 - ANN trained on classes c2
# Output:  ann_output_model - Consolidated ANN using Weight Summation Technique on classes (c1 U c2)
def perform_weight_summation_c10(model1, model2, ann_base_model):
    # Get MetaData for Model1
    layer_weights_1 = {}
    layer_biases_1 = {}
    for layer_number in range(len(model1.layers)):
        if layer_number == 2 or layer_number == 3 or layer_number == 5 or layer_number == 7 or layer_number == 8 or layer_number == 9:  # Skip
            # weight collection for
            # Pooling and Flatten Layers
            continue

        layer_weights_1[layer_number] = model1.layers[layer_number].get_weights()[0]
        layer_biases_1[layer_number] = model1.layers[layer_number].get_weights()[1]

    # Get MetaData for Model2
    layer_weights_2 = {}
    layer_biases_2 = {}
    for layer_number in range(len(model2.layers)):
        if layer_number == 2 or layer_number == 3 or layer_number == 5 or layer_number == 7 or layer_number == 8 or layer_number == 9:  # Skip weight collection for Pooling and Flatten Layers
            continue
        layer_weights_2[layer_number] = model2.layers[layer_number].get_weights()[0]
        layer_biases_2[layer_number] = model2.layers[layer_number].get_weights()[1]

    ann_output_model = create_cnn_model_c10()
    for layer_number in range(len(ann_output_model.layers)):
        if layer_number == 2 or layer_number == 3 or layer_number == 5 or layer_number == 7 or layer_number == 8 or layer_number == 9: # Skip weight aggregation for Pooling and Flatten Layers
            continue

        if layer_number == 0 or layer_number == 1 or layer_number == 4 or layer_number == 6 or layer_number == 10:
            ann_output_model.layers[layer_number].set_weights(ann_base_model.layers[layer_number].get_weights())
            continue
        #ann_output_model.layers[layer_number].set_weights([np.multiply(layer_weights_1[layer_number], layer_weights_2[layer_number]),
        #                                                   np.multiply(layer_biases_1[layer_number],layer_biases_2[layer_number])])
        ann_output_model.layers[layer_number].set_weights([(layer_weights_1[layer_number]+ layer_weights_2[layer_number])/2.0,
                                                           (layer_biases_1[layer_number]+layer_biases_2[layer_number])/2.0])

    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    ann_output_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return ann_output_model

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

    ann_output_model = create_cnn_model_usps()
    for layer_number in range(len(ann_output_model.layers)):
        if layer_number == 1 or layer_number == 2:  # Skip weight aggregation for Pooling and Flatten Layers
            continue
        ann_output_model.layers[layer_number].set_weights([(layer_weights_1[layer_number]+ layer_weights_2[layer_number]),
                                                           (layer_biases_1[layer_number]+layer_biases_2[layer_number])])

    return ann_output_model

def perform_mnist_operations():
    # Get MNIST dataset
    X_train, Y_train, X_test, Y_test = load_dataset_mnist()

    # Split Training Data and Test data for Different Data Models
    X_train1, Y_train1, X_train2, Y_train2, X_test1, Y_test1, X_test2, Y_test2 = split_train_test_data(X_train, Y_train, X_test, Y_test)

    # Get Trained Models for network1 and network2
    ann_model1, ann_model2 = train_cnn_models(X_train1, Y_train1, X_train2, Y_train2)

    visualize_model(ann_model1,X_train1)
    # Build Consolidated ANN using Weight Summation Technique
    #ann_output_model = perform_weight_summation(ann_model1, ann_model2)

    # Get Accuracy of Consolidated Model on test data
    #get_accuracy(ann_output_model,np.append(X_test1,X_test2,axis=0),np.append(Y_test1,Y_test2,axis=0))


def get_cifar10_training_testdata():
    X_training = np.zeros(shape=(1,32,32,3))
    Y_training = np.zeros(shape=(1))
    X_test = X_training
    Y_test = Y_training

    # Get CIFAR dataset
    for batch_number in range(1,6):
        X,Y = load_cifar10('CIFAR10_Dataset/cifar-10-python/cifar-10-batches-py/data_batch_'+str(batch_number))
        X = X.astype('float32')/255.0
        X_training = np.append(X_training,X,axis=0)
        Y_training = np.append(Y_training,Y)
    X_training = X_training[1:,:,:,:]
    Y_training = Y_training[1:]

    # Get CIFAR testset
    X_test,Y_test = load_cifar10('CIFAR10_Dataset/cifar-10-python/cifar-10-batches-py/test_batch')
    X_test = X_test.astype('float32')/255.0

    return X_training,Y_training,X_test,Y_test

def perform_cifar10_operations():

    # Get training data
    X_train,Y_train,X_test,Y_test = get_cifar10_training_testdata()

    # Split Training Data
    X_train1, Y_train1, X_train2, Y_train2, X_test1, Y_test1, X_test2, Y_test2 = split_train_test_data_c10(X_train,Y_train,X_test,Y_test)

    # Get Trained Models for network1 and network2
    ann_full_output = train_complete_model(np.append(X_train1,X_train2,axis=0),np.append(Y_train1,Y_train2,axis=0))

    ann_model1, ann_model2 = train_cnn_models(X_train1, Y_train1, X_train2, Y_train2,ann_full_output)

    # Build Consolidated ANN using Weight Summation Technique
    ann_output_model = perform_weight_summation_c10(ann_model1, ann_model2, ann_full_output)


    # Get Accuracy of Consolidated Model on test data
    #get_accuracy(ann_output_model,X_train1,Y_train1)
    get_accuracy(ann_output_model,np.append(X_test1,X_test2,axis=0),np.append(Y_test1,Y_test2,axis=0))
    #get_accuracy(ann_output_model,X_train2,Y_train2)
    get_accuracy_labelwise(ann_output_model,np.append(X_test1,X_test2,axis=0),np.append(Y_test1,Y_test2,axis=0))

def visualize_model(model,X_train):

    print(model.summary())

    model = Model(inputs=model.inputs,outputs=model.layers[0].output)
    sample = X_train[2].reshape(1,28,28,1)
    feature_maps = model.predict(sample)
    side1 = 8
    side2 = 4
    ix = 1
    for _ in range(side1):
        for _ in range(side2):
            # specify subplot and turn of axis
            ax = plt.subplot(side1, side2, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()

def perform_usps_operations():
    X_train,Y_train,X_test,Y_test = load_USPS()

    # Split Training Data and Test data for Different Data Models
    X_train1, Y_train1, X_train2, Y_train2, X_test1, Y_test1, X_test2, Y_test2 = split_train_test_data_c10(X_train, Y_train, X_test, Y_test)

    # Get Trained Models for network1 and network2
    ann_model1, ann_model2 = train_cnn_models(X_train1, Y_train1, X_train2, Y_train2)

    # Build Consolidated ANN using Weight Summation Technique
    ann_output_model = perform_weight_summation(ann_model1, ann_model2)

    # Get Accuracy of Consolidated Model on test data
    get_accuracy(ann_output_model,np.append(X_test1,X_test2,axis=0),np.append(Y_test1,Y_test2,axis=0))
    get_accuracy(ann_output_model,np.append(X_train1,X_train2,axis=0),np.append(Y_train1,Y_train2,axis=0))



# Main Function
def main():

    perform_cifar10_operations()

    #perform_mnist_operations()

    #perform_usps_operations()


if __name__ == "__main__":
    main()
