import os


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

# Set Directory Path
DATA_DIR = os.getcwd() + "/Images/"

# -------------------------------------------------------------------------------------------

data = os.listdir(DATA_DIR)  # 8680

train_size = int(0.8 * len(data))

train = data[:train_size]
train_len = data_split(train, a='train')  # 6944

test = data[train_size:]
test_len = data_split(test, a='test')  # 1736

print("Train and Test ids files have been created successfully.")
# -------------------------------------------------------------------------------------------
