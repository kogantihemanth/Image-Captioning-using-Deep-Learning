import os
os.system("pip3 install googledrivedownloader")
os.system("pip3 install requests")
from google_drive_downloader import GoogleDriveDownloader as gdd

# to download Images
gdd.download_file_from_google_drive(file_id='1g4ExCFgRTOA6BcO1BMHSfh7u4QpNHM5t',
                                            dest_path='./images.zip',
                                            unzip=True)

# to download glove
gdd.download_file_from_google_drive(file_id='1OMm6HSwSvn-PrYNRGmX6ajQA-mWc-wOr',
                                            dest_path='./glove.zip',
                                            unzip=True)

# to download captions
gdd.download_file_from_google_drive(file_id='1K_c-R2MIBGX9cR7A5N-KozhmoeJ0hBbl',
                                            dest_path='./captions.zip',
                                            unzip=True)
