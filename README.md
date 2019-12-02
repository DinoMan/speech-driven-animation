# Speech-Driven Animation

This library implements the end-to-end facial synthesis model described in this [paper](https://sites.google.com/view/facialsynthesis/home).

![speech-driven-animation](example.gif)

## Prerequisites
The models provided are checked out using git LFS. You can install git LFS by following these [instructions](https://help.github.com/en/articles/installing-git-large-file-storage).

## Downloading the models
The models were hosted on git LFS. However the demand was so high that I reached the quota for free gitLFS storage. I have moved the models to GoogleDrive. Models can be found [here](https://drive.google.com/open?id=1pJdsnknLmMLvA8RQIAV3AQH8vU0FeK16).

## Installing

To install the library do:
```
$ pip install .
```

## Running the example

To create the animations you will need to instantiate the VideoAnimator class. Then you provide an image and audio clip (or the paths to the files) and a video will be produced.


## Choosing the model
The model has been trained on the GRID, TCD-TIMIT, CREMA-D and LRW datasets. The default model is GRID. To load another pretrained model simply instantiate the VideoAnimator with the following arguments:

```
import sda
va = sda.VideoAnimator(gpu=0, model_path="crema")# Instantiate the animator
```

The models that are currently uploaded are:
- [x] grid
- [x] timit
- [x] crema
- [ ] lrw


### Example with Image and Audio Paths
```
import sda
va = sda.VideoAnimator(gpu=0)# Instantiate the animator
vid, aud = va("example/image.bmp", "example/audio.wav")
```

### Example with Numpy Arrays
```
import sda
import scipy.io.wavfile as wav
from PIL import Image

va = sda.VideoAnimator(gpu=0)# Instantiate the animator
fs, audio_clip = wav.read("example/audio.wav")
still_frame = Image.open("example/image.bmp")
vid, aud = va(frame, audio_clip, fs=fs)
```

### Saving video with audio
```
va.save_video(vid, aud, "generated.mp4")
```

## Using the encodings
The encoders for audio and video are made available so that they can be used to produce features for classification tasks.

### Audio Encoder
The Audio Encoder (which is made of Audio-Frame encoder and RNN) is provided along with a dictionary which has information such as the feature length (in seconds) required by the Audio Frame encoder and the overlap between audio frames.
```
import sda
encoder, info = sda.get_audio_feature_extractor(gpu=0)
```
