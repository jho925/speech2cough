import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import numpy as np
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", help="train file",default= "train.csv")
parser.add_argument("--test_file", help="test file",default= "test.csv")
parser.add_argument("--data_path", help="data_path",default= "data/")
args = parser.parse_args()

def audio_to_spectrogram(audio,img):
  plt.rcParams["figure.figsize"] = [2.56, 2.56]
  plt.rcParams["figure.autolayout"] = True

  fig, ax = plt.subplots()


  y, sr = librosa.load(audio)
  S = librosa.feature.melspectrogram(y=y, sr=sr)
  S_dB = librosa.power_to_db(S, ref=np.max)
  p = librosa.display.specshow(S_dB, sr=sr, fmax=8000, ax=ax)
  print(S.shape)

  plt.savefig(img)

# load all images in a directory into memory
def load_images(csv, size=(256,256)):
  df = pd.read_csv(csv)
  speech_list, cough_list = list(), list()

  for index, row in df.iterrows():
    speech_pixels = load_img(args.data_path + row['speech_path'][:-4] + '.png', target_size=size)
    speech_pixels = img_to_array(speech_pixels)

    cough_pixels = load_img(args.data_path + row['cough_path'][:-4] + '.png', target_size=size)
    cough_pixels = img_to_array(cough_pixels)


    cough_list.append(cough_pixels)
    speech_list.append(speech_pixels)

  return [asarray(speech_list), asarray(cough_list)]


def main():
  df = pd.read_csv(args.train_file)
  df1 = pd.read_csv(args.test_file)
  conversion = df['cough_path'].tolist() + df[speech_path].tolist() + df1['cough_path'].tolist() + df1[speech_path].tolist()

  for i in conversion:
    audio_to_spectrogram(args.data_path + i,args.data_path + i[:-4] + 'png')

  [speech_images, cough_images] = load_images(args.train_file)
  print('Loaded: ', speech_images.shape, cough_images.shape)
  # save as compressed numpy array
  filename = 'cough2speech.npz'
  savez_compressed(filename, speech_images, cough_images)
  print('Saved dataset: ', filename)
  


if __name__ == '__main__':
    main()