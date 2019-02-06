# Speech-Driven Animation

This library implements the end-to-end facial synthesis model described in this [paper](https://sites.google.com/view/facialsynthesis/home).

![speech-driven-animation](example.jpg)

## Installing

To install the library do:
```
$ pip install .
```

## Running the example

To create the animations you will need to instantiate the VideoAnimator class. Then you provide an image and audio clip (or the paths to the files) and a video will be produced.

### Example with Image and Audio Paths
```
import sda
va = sda.VideoAnimator(gpu=0)# Instantiate the aminator
vid, aud = va("example/image.bmp", "example/audio.wav")
```

### Example with Image and Audio Paths
```
import sda
import scipy.io.wavfile as wav
from PIL import Image

va = sda.VideoAnimator(gpu=0)# Instantiate the aminator
fs, audio_clip = wav.read("example/audio.wav")
still_frame = Image.open("example/image.bmp")
vid, aud = va(frame, audio_clip, fs=fs)
```

### Saving video with audio
```
va.save_video(vid, aud, "/home/SERILOCAL/k.vougioukas/jfk.mp4")
```
