import numpy as np
import os
import string
import glob
from keras.models import load_model
from pickle import dump, load
from keras.applications.nasnet import NASNetLarge
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.nasnet import preprocess_input
from keras.utils import plot_model
from keras.models import Model

# ------------------------------------------------------------------------------------

# Loading files from disk
images = os.getcwd() + "/Images/"
captions = os.getcwd() + "/captions.txt"
train_imgs = os.getcwd() + "/train_ids.txt"
test_imgs = os.getcwd() + "/test_ids.txt"

# ------------------------------------------------------------------------------------

# TEXT PRE-PROCESSING

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load descriptions
doc = load_doc(captions)
# print(doc[:300])


def load_descriptions(doc):
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping


# parse descriptions
descriptions = load_descriptions(doc)
# print(descriptions)
print('Loaded: %d ' % len(descriptions))


def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] =  ' '.join(desc)


# clean descriptions
# clean = clean_descriptions(descriptions)
# print(clean)
clean_descriptions(descriptions)
# print(descriptions)


# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


# summarize vocabulary
vocabulary = to_vocabulary(descriptions)
# print(vocabulary)
print('Original Vocabulary Size: %d' % len(vocabulary))


# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


save_descriptions(descriptions, 'descriptions.txt')


# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# load training dataset (6K)
train = load_set(train_imgs)
# print(train)
print('Dataset: %d' % len(train))


# Below path contains all the images
image_all = images
# Create a list of all image names in the directory
img = glob.glob(image_all + '*.jpg')


# Below file contains the names of images to be used in train data
train_images_file = 'train_ids.txt'
# Read the train image names in a set
train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

# Create a list of all the training images with their full path names
train_img = []

for i in img: # img is list of full path names of all images
    if i[len(images):] in train_images: # Check if the image belongs to training set
        train_img.append(i) # Add it to the list of train images


# Below file conatains the names of images to be used in test data
test_images_file = 'test_ids.txt'
# Read the validation image names in a set# Read the test image names in a set
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

# Create a list of all the test images with their full path names
test_img = []

for i in img: # img is list of full path names of all images
    if i[len(images):] in test_images: # Check if the image belongs to test set
        test_img.append(i) # Add it to the list of test images


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions

# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# print(train_descriptions)

# train_features = load(open("features_8Ktrain.pkl", "rb"))
# print('Photos: train=%d' % len(train_features))

# Create a list of all the training captions
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
print('Length of all train captions :',len(all_train_captions))

# Consider only words which occur at least 10 times in the corpus
word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('Preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1  # one for appended 0's
print('Vocab size :',vocab_size)

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# print(to_lines(descriptions))

# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# determine the maximum sequence length
max_len = max_length(train_descriptions)
print('Description Length: %d' % max_len)

# ------------------------------------------------------------------------------------

# FEATURE EXTRACTION USING ResNet50

# load a pre-defined list of photo identifiers
def load_list(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return dataset


# extract features from each photo in the directory
def extract_features(directory, ids, model):
    if int(model) == 1:
        print("1")
        # load ResNet50 model
        model = ResNet50()
        input_size = 224
    else:
        print("2")
        # load NASNetLarge model
        model = NASNetLarge(input_shape=(331, 331, 3), include_top=True, weights='imagenet', input_tensor=None, pooling=None)
        input_size = 331
    # pops the last layer to get the features
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    model.summary()
    print(len(model.layers))
    # model characteristics
    plot_model(model, to_file='model.png')
    imgs = load_list(ids)
    print('Dataset: %d' % len(imgs))
    N = len(imgs)
    print(N)
    results = []
    i = 0
    batch_size = 1  # this can be 8 for a GTX 1080 Ti and 32G of RAM
    while i < N:
        if i % 1024 == 0:
            print('{} from {} images.'.format(i, N))
        batch = imgs[i:i + batch_size]
        i += batch_size
        images = [
            load_img(
                os.path.join(directory, img + ".jpg"),
                target_size=(input_size, input_size)
            )
            for img in batch
        ]
        images = [preprocess_input(img_to_array(img)) for img in images]
        images = np.stack(images)
        r = model.predict(images)
        for ind in range(batch_size):
            results.append(r[ind])
    return results


imgs_directory = images
imgs_ids = test_imgs
model = 1   # 1 - ResNet50, 2 - NasnetLarge
features = extract_features(imgs_directory, imgs_ids, model)
file = open("features_resnet_8K.pkl", "wb")
dump(features, file)

model2 = 2
features2 = extract_features(imgs_directory, imgs_ids, model2)
file = open("features_nasnet_8K.pkl", "wb")
dump(features2, file)

imgs_ids_tr = train_imgs
model_t1 = 1   # 1 - ResNet50, 2 - NasnetLarge
features = extract_features(imgs_directory, imgs_ids_tr, model_t1)
file = open("features_resnet_train.pkl", "wb")
dump(features, file)

model_t2 = 2
features2 = extract_features(imgs_directory, imgs_ids_tr, model_t2)
file = open("features_nasnet_train.pkl", "wb")
dump(features2, file)