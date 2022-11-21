# Speech-Driven Animation

This library implements the end-to-end facial synthesis model described in this [paper](https://sites.google.com/view/facialsynthesis/home).

This library is maintained by Konstantinos Vougioukas, Honglie Chen and Pingchuan Ma.

![speech-driven-animation](example.gif)

## Downloading the models
The models were hosted on git LFS. However the demand was so high that I reached the quota for free gitLFS storage. I have moved the models to GoogleDrive. Models can be found [here](https://drive.google.com/drive/folders/17Dc2keVoNSrlrOdLL3kXdM8wjb20zkbF?usp=sharing).
Place the model file(s) under *`sda/data/`*

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
va = sda.VideoAnimator(gpu=0, model_path="crema")  # Instantiate the animator
```

The models that are currently uploaded are:
- [x] GRID
- [x] TIMIT
- [x] CREMA
- [ ] LRW


### Example with image and audio paths
```
import sda
va = sda.VideoAnimator(gpu=0)  # Instantiate the animator
vid, aud = va("example/image.bmp", "example/audio.wav")
```

### Example with numpy arrays
```
import sda
import scipy.io.wavfile as wav
from PIL import Image

va = sda.VideoAnimator(gpu=0) # Instantiate the animator
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

### Audio encoder
The Audio encoder (which is made of Audio-frame encoder and RNN) is provided along with a dictionary which has information such as the feature length (in seconds) required by the Audio Frame encoder and the overlap between audio frames.
```
import sda
encoder, info = sda.get_audio_feature_extractor(gpu=0)
```

## Citation

If you find this code useful in your research, please consider to cite the following papers:

```bibtex
@inproceedings{vougioukas2019end,
  title={End-to-End Speech-Driven Realistic Facial Animation with Temporal GANs.},
  author={Vougioukas, Konstantinos and Petridis, Stavros and Pantic, Maja},
  booktitle={CVPR Workshops},
  pages={37--40},
  year={2019}
}
```
