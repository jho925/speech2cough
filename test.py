import pandas as pd
from numpy import load
from matplotlib import pyplot
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot as plt
import tensorflow as tf
from PIL import Image 
import PIL
import numpy as np
import librosa
from numpy import asarray
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument("--test_file", help="test file",default= "test.csv")
parser.add_argument("--model_path", help="model path", default = 'model.h5')
parser.add_argument("--data_path", help="data_path",default= "data/")
parser.add_argument("--results_path", help="results path", default = 'results/')
args = parser.parse_args()

# load an image
def load_image(filename, size=(256,256)):
    # load image with the preferred size
    pixels = load_img(filename, target_size=size)
    # convert to numpy array
    pixels = img_to_array(pixels)
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    # reshape to 1 sample
    pixels = expand_dims(pixels, 0)
    return pixels

def reconstruct_signal(S_db, ref=1.0):
    """Builds an audio signal (numpy array) from a spectogram."""
    sample_rate = 44100
    n_fft = 2048
    hop_length = 518
    
    S = librosa.db_to_power(S_db, ref=ref)
    
    audio = librosa.feature.inverse.mel_to_audio(
        M=S,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return audio

def get_filenames(csv):
    """Gather all files corresponding to source and target.
    Put them in a list of tuples.
    """
    df = pd.read_csv(csv)
    speech_path = df['speech_path'].tolist() 
    cough_path = df['cough_path'].tolist()


    return list(zip(speech_path, cough_path))


def reconstruct_signal(S_db, ref=1.0):
    """Builds an audio signal (numpy array) from a spectogram."""
    sample_rate = 44100
    n_fft = 2048
    hop_length = 518
    
    S = librosa.db_to_power(S_db, ref=ref)
    
    audio = librosa.feature.inverse.mel_to_audio(
        M=S,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return audio

        



def main():
    df = pd.read_csv(args.test_file)

    test_speech = df['speech_path'].tolist

    

    model = load_model(args.model_path)

    for i in test_speech:
        # load source image
        
        gen_image = model.predict(args.data_path + i[:-4] + 'png')
        # # scale from [-1,1] to [0,1]
        gen_image = (gen_image + 1) / 2.0

        pred_img = load_img(gen_image[0])
        img_pixels = img_to_array(pred_img)
        aud = reconstruct_signal(img_pixels)


        sf.write(args.results_path + 'predict_' + i, aud, 44100)

        plt.imshow(gen_image[0])
        plt.axis('off')
        plt.savefig(args.results_path + i[:-4] + '.png')
    




if __name__ == '__main__':
    main()