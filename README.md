# Final-Project-Group6
###  Data source link: https://www.kaggle.com/adityajn105/flickr8k/download
#### The data is downloaded and included in the repository.
### glove.6B.200d.txt: https://www.kaggle.com/incorpes/glove6b200d/download
####  This is the pretrained glove model has 200-dimensional vector.
##  Steps to run the code:
### Step 1: Run data_download.py
#### Downloads Images, Captions and Glove.txt and unzips the folders and files.
### Step 1: Unzip the glove.6B.200d txt file and place it in the Final-Project-Group6/Code/ folder
### Step 2: Download the repository
### Step 3: Go to the Final-Project-Group6/Code/ folder
### Step 4: Run DataSplit.py file 
####    This generates train_ids.txt & test_ids.txt files which are also included in the folder so this step is optional.
### Step 5: Run feature_generator.py file
####    This generates train_features8k_vgg16.pkl & train_features8k_vgg16.pkl files which are also included in the folder so this step is optional.
### Step 6: Run modeling.py file.
