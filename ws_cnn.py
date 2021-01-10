import gzip
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
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
def print_image(image_data):
    plt.imshow(image_data)
    plt.show()

# Split Training Dataset based on numbers
def split_train_data(X_train,Y_train):

    split_category1 = [0,1,2,3,4,9]
    split_category2 = [5,6,7,8]

    X_train1 = np.array([X_train[i,0] for i, x in enumerate(Y_train) if x in split_category1])
    Y_train1 = np.array([x for i, x in enumerate(Y_train) if x in split_category1])

    X_train2 = np.array( [X_train[i,0] for i, x in enumerate(Y_train) if x in split_category2])
    Y_train2 = np.array([x for i, x in enumerate(Y_train) if x in split_category2])

    return X_train1,Y_train1,X_train2,Y_train2

# Create Neural Networks from keras package
def create_cnn_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))
    model.add(Dense(32))

    return model

def main():
    # Get MNIST dataset
    X_train, Y_train, X_test, Y_test = load_dataset_mnist()

    # Split Training Data for Different Data Models
    X_train1,Y_train1,X_train2,Y_train2 = split_train_data(X_train,Y_train)

    create_cnn_model()


if __name__ == "__main__":
    main()
