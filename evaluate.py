import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--test_file", help="test file",default= "test.csv")
parser.add_argument("--results_path", help="results path", default = 'results/')
parser.add_argument("--data_path", help="data_path",default= "data/")
args = parser.parse_args()

def mse(imageA, imageB):
   # the 'Mean Squared Error' between the two images is the
   # sum of the squared difference between the two images;
   # NOTE: the two images must have the same dimension
   err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
   err /= float(imageA.shape[0] * imageA.shape[1])
   
   # return the MSE, the lower the error, the more "similar"
   # the two images are
   return err


def compare_images(imageA, imageB, title):
   # compute the mean squared error and structural similarity
   # index for the images
   m = mse(imageA, imageB)
   s = ssim(imageA, imageB, channel_axis=2)
   # setup the figure
   fig = plt.figure(title)
   plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
   # show first image
   ax = fig.add_subplot(1, 2, 1)
   plt.imshow(imageA, cmap = plt.cm.gray)
   plt.axis("off")
   # show the second image
   ax = fig.add_subplot(1, 2, 2)
   plt.imshow(imageB, cmap = plt.cm.gray)
   plt.axis("off")
   # show the images
   plt.show()

def main():
   # load the input images
   df = pd.read_csv(args.test_file)
   total_ssim = 0 
   for index, row in df.iterrows():

      predict_cough = cv2.imread(args.results_path + row['speech_path'][:-4] + '.png')
      cough = cv2.imread(args.data_path +  row['cough_path'][:-4] + '.png')

      # define the function to compute MSE between two image

      total_ssim += ssim(cough, predict_cough, channel_axis=2)
   

   print(total_ssim/(index +1))

if __name__ == '__main__':
   main()