"""
This code is partially borrowed from 
https://github.com/shaohua0116/WGAN-GP-TensorFlow/blob/master/download.py
"""
import os
import glob
import argparse
import numpy as np
import subprocess
from zipfile import ZipFile

parser = argparse.ArgumentParser(description='Download CelebA dataset and split it into train and test set')
parser.add_argument('--download_path', type=str, default='./data')
PARAMS = parser.parse_args()

def download_file_from_google_drive(id, destination):
    import requests
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def download_celeba(download_path):
    data_dir = os.path.join(download_path, 'train')
    
    if os.path.exists(data_dir):
        print('celeba was downloaded')
        return
    else:
        os.makedirs(data_dir)
    

    cmds = [['unzip', 'celebA'], ['rm', 'celebA'],
            ['mv', 'img_align_celeba', 'train'], ['mv', 'train', download_path],
            ['mkdir', download_path+'/test']]
    for cmd in cmds:
        subprocess.call(cmd)
    

print('Downloading celeba')
#download_file_from_google_drive('0B7EVK8r0v71pTUZsaXdaSnZBZzg', 'celebA.zip')
download_file_from_google_drive("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "celebA")
print('Unzipping ')

download_celeba(PARAMS.download_path)
subprocess.call("mv {}/train/2025* {}/test".format(PARAMS.download_path, PARAMS.download_path), shell=True)
