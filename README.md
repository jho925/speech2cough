# speech2cough

## Installation

Proceed below to complete the installation

1. `git clone https://github.com/jho925/speech2cough.git`
2. `cd speech2cough`

## Dataset
Download the FluSense Dataset

* FluSense Dataset, “A large-scale and high-quality dataset of annotated musical notes.” 
https://github.com/Forsad/FluSense-data

## Preprocessing

Preprocess the training data and generate the spectrograms using the scripts below

```
$ python preprocessing.py --train_file <TRAIN_FILE>
                          --test_file <TEST_FILE>
                          --data_path <DATA_PATH>
```

Where TRAIN_FILE and TEST_FILE are CSVs with column labels 'speech_path' and 'cough_path', and DATA_PATH is the folder containing the audio files

## Training
Train the model using the scripts below

```
$ python main.py
```

Models are saved every 10 epochs and saved to a file with the training iteration number. Additionally, images are generated every 10 epochs and compared to the expected target images

## Test
Test the model using the scripts below

```
$ python test.py --test_file <TEST_FILE>
			     --model_path <MODEL_PATH>
                 --results_path <RESULTS_PATH>
                 --data_path <DATA_PATH>
```
Where RESULTS_PATH is the folder in which you would like your results to be stored, and MODEL_PATH is the path to the model checkpoint

## Evaluate

Evaluate the results according the Structural Similarity Index (SSIM) using the scripts below

```
$ python evaluate.py --test_file <TEST_FILE>
                     --results_path <RESULTS_PATH>
                     --data_path <DATA_PATH>
```

## License

Distributed under the MIT License. See license.txt for more information.
