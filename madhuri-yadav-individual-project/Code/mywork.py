
import os
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet_v2 import ResNet50V2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import pickle


# -------------------------------------------------------------------------------------------

# Function to Split Data in Train set (80%) and Test set (20%)
# Create text files with training and testing ids

def data_split(data, a=''):
    f = open('{}_ids.txt'.format(a), 'w+')
    for i in range(len(data)):
        f.write(data[i] + '\n')
    f.close()
    f = open('{}_ids.txt'.format(a), 'r')
    f1 = f.readlines()
    return len(f1)


# -------------------------------------------------------------------------------------------

# extract features from each photo in the directory
def extract_features(directory, textfile, m):
    if m == 1:
        # load the model
        model = VGG16()
        # re-structure the model
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        # summarize
        print(model.summary())
        m = "vgg16"
    elif m == 2:
        # load the model
        model = InceptionV3(weights='imagenet')
        # re-structure the model
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        # summarize
        print(model.summary())
        m = "IV3"
    else:
        # load the model
        model = ResNet50V2()
        # re-structure the model
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        # summarize
        print(model.summary())
        m = "ResNet50V2"

    # Read Train ids as a list for feature extraction
    with open(textfile) as f:
        train_ids = f.read().splitlines()
    # extract features from each photo
    features = dict()
    for name in train_ids:
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        print('>%s' % name)
    return features, m


# -------------------------------------------------------------------------------------------
# Set Directory Path
DATA_DIR = os.getcwd() + "/Images"
# -------------------------------------------------------------------------------------------

data = os.listdir(DATA_DIR)  # 8680

train_size = int(0.8 * len(data))

train = data[:train_size]
train_len = data_split(train, a='train')  # 6944

test = data[train_size:]
test_len = data_split(test, a='test')  # 1736

# -------------------------------------------------------------------------------------------
# 1 for VGG16,  2 for InceptionV3,  3 for ResNet50V2
features, m = extract_features(DATA_DIR, 'train_ids.txt', 1)  # (output size 4096)
trainfile = "train_features8k_" + m + ".pkl"
file = open(trainfile, "wb")
dump(features, file)
file.close()

features, m = extract_features(DATA_DIR, 'test_ids.txt', 1)  # (output size 4096)
testfile = "test_features8k_" + m + ".pkl"
file = open(testfile, "wb")
dump(features, file)
file.close()
# -------------------------------------------------------------------------------------------

file_to_read = open(trainfile, "rb")
loaded_dictionary = pickle.load(file_to_read)
print(loaded_dictionary)

file_to_read = open(testfile, "rb")
loaded_dictionary = pickle.load(file_to_read)
print(loaded_dictionary)
