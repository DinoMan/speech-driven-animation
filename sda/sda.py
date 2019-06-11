from torchvision import transforms
import torch
import torchaudio
from .encoder_image import Encoder
from .img_generator import Generator
from .rnn_audio import RNN

from scipy import signal
import numpy as np
from PIL import Image
import contextlib
import os
import shutil
import tempfile
import skvideo.io as sio
import scipy.io.wavfile as wav
import ffmpeg


@contextlib.contextmanager
def cd(newdir, cleanup=lambda: True):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        cleanup()


@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()

    def cleanup():
        shutil.rmtree(dirpath)

    with cd(dirpath, cleanup):
        yield dirpath


def get_audio_feature_extractor(model_path=None, gpu=-1):
    if model_path is None:
        model_path = os.path.split(__file__)[0] + "/data/model.dat"

    if gpu < 0:
        device = torch.device("cpu")
        model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.device("cuda:" + str(gpu))
        model_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpu))

    audio_rate = model_dict["audio_rate"]
    audio_feat_len = model_dict['audio_feat_len']
    rnn_gen_dim = model_dict['rnn_gen_dim']
    aud_enc_dim = model_dict['aud_enc_dim']
    video_rate = model_dict["video_rate"]

    encoder = RNN(audio_feat_len, aud_enc_dim, rnn_gen_dim, audio_rate, init_kernel=0.005, init_stride=0.001)
    encoder.to(device)
    encoder.load_state_dict(model_dict['encoder'])

    overlap = audio_feat_len - 1.0 / video_rate
    return encoder, {"rate": audio_rate, "feature length": audio_feat_len, "overlap": overlap}


def cut_audio_sequence(seq, feature_length, overlap, rate):
    seq = seq.view(-1, 1)
    snip_length = int(feature_length * rate)
    cutting_stride = int((feature_length - overlap) * rate)
    pad_samples = snip_length - cutting_stride

    pad_left = torch.zeros(pad_samples // 2, 1, device=seq.device)
    pad_right = torch.zeros(pad_samples - pad_samples // 2, 1, device=seq.device)

    seq = torch.cat((pad_left, seq), 0)
    seq = torch.cat((seq, pad_right), 0)

    stacked = seq.narrow(0, 0, snip_length).unsqueeze(0)
    iterations = (seq.size()[0] - snip_length) // cutting_stride + 1
    for i in range(1, iterations):
        stacked = torch.cat((stacked, seq.narrow(0, i * cutting_stride, snip_length).unsqueeze(0)))
    return stacked


class VideoAnimator():
    def __init__(self, model_path=None, gpu=-1):

        if model_path is None:
            model_path = os.path.split(__file__)[0] + "/data/model.dat"

        if gpu < 0:
            self.device = torch.device("cpu")
            model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        else:
            self.device = torch.device("cuda:" + str(gpu))
            model_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpu))

        self.img_size = model_dict["img_size"]
        self.audio_rate = model_dict["audio_rate"]
        self.video_rate = model_dict["video_rate"]
        self.audio_feat_len = model_dict['audio_feat_len']
        self.audio_feat_samples = model_dict['audio_feat_samples']
        self.id_enc_dim = model_dict['id_enc_dim']
        self.rnn_gen_dim = model_dict['rnn_gen_dim']
        self.aud_enc_dim = model_dict['aud_enc_dim']
        self.aux_latent = model_dict['aux_latent']
        self.sequential_noise = model_dict['sequential_noise']

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size[0], self.img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.audio_transform = torchaudio.transforms.Scale()

        self.encoder = RNN(self.audio_feat_len, self.aud_enc_dim, self.rnn_gen_dim,
                           self.audio_rate, init_kernel=0.005, init_stride=0.001)
        self.encoder.to(self.device)
        self.encoder.load_state_dict(model_dict['encoder'])

        self.encoder_id = Encoder(self.id_enc_dim, self.img_size)
        self.encoder_id.to(self.device)
        self.encoder_id.load_state_dict(model_dict['encoder_id'])

        skip_channels = list(self.encoder_id.channels)
        skip_channels.reverse()

        self.generator = Generator(self.img_size, self.rnn_gen_dim, condition_size=self.id_enc_dim,
                                   num_gen_channels=self.encoder_id.channels[-1],
                                   skip_channels=skip_channels, aux_size=self.aux_latent,
                                   sequential_noise=self.sequential_noise)

        self.generator.to(self.device)
        self.generator.load_state_dict(model_dict['generator'])

        self.encoder.eval()
        self.encoder_id.eval()
        self.generator.eval()

    def save_video(self, video, audio, path, experimental_ffmpeg=False):
        with tempdir() as dirpath:
            # Save the video file
            writer = sio.FFmpegWriter(dirpath + "/tmp.avi",
                                      inputdict={'-r': str(self.video_rate) + "/1", },
                                      outputdict={'-r': str(self.video_rate) + "/1", }
                                      )
            for i in range(video.shape[0]):
                writer.writeFrame(np.rollaxis(video[i, :, :, :], 0, 3))
            writer.close()

            # Save the audio file
            wav.write(dirpath + "/tmp.wav", self.audio_rate, audio)

            in1 = ffmpeg.input(dirpath + "/tmp.avi")
            in2 = ffmpeg.input(dirpath + "/tmp.wav")
            if experimental_ffmpeg:
                out = ffmpeg.output(in1['v'], in2['a'], path, strict='-2', loglevel="panic")
            else:
                out = ffmpeg.output(in1['v'], in2['a'], path, loglevel="panic")
            out.run()

    def _cut_sequence_(self, seq, cutting_stride, pad_samples):
        pad_left = torch.zeros(pad_samples // 2, 1)
        pad_right = torch.zeros(pad_samples - pad_samples // 2, 1)

        seq = torch.cat((pad_left, seq), 0)
        seq = torch.cat((seq, pad_right), 0)

        stacked = seq.narrow(0, 0, self.audio_feat_samples).unsqueeze(0)
        iterations = (seq.size()[0] - self.audio_feat_samples) // cutting_stride + 1
        for i in range(1, iterations):
            stacked = torch.cat((stacked, seq.narrow(0, i * cutting_stride, self.audio_feat_samples).unsqueeze(0)))
        return stacked.to(self.device)

    def _broadcast_elements_(self, batch, repeat_no):
        total_tensors = []
        for i in range(0, batch.size()[0]):
            total_tensors += [torch.stack(repeat_no * [batch[i]])]

        return torch.stack(total_tensors)

    def __call__(self, img, audio, fs=None):
        if isinstance(img, str):  # if we have a path then grab the image
            frame = Image.open(img)
        else:
            frame = img

        if isinstance(audio, str):  # if we have a path then grab the image
            speech, audio_rate = torchaudio.load(audio, channels_first=False)
            if speech.size(1) != 1:
                speech = speech[:, 0].view(-1, 1)
        else:
            if fs is None:
                raise AttributeError("Audio provided without specifying the rate. Specify rate or use audio file!")
            seq_length = audio.shape[0]
            max_value = np.iinfo(audio.dtype).max
            speech = torch.from_numpy(
                2 * signal.resample(audio, int(seq_length * self.audio_rate / fs)) / max_value).float()
            speech = speech.view(-1, 1)

        frame = self.img_transform(frame).to(self.device)

        cutting_stride = int(self.audio_rate / self.video_rate)
        audio_seq_padding = self.audio_feat_samples - cutting_stride

        # Create new sequences of the audio windows
        audio_feat_seq = self._cut_sequence_(speech, cutting_stride, audio_seq_padding)
        frame = frame.unsqueeze(0)
        audio_feat_seq = audio_feat_seq.unsqueeze(0)
        audio_feat_seq_length = audio_feat_seq.size()[1]

        z = self.encoder(audio_feat_seq, [audio_feat_seq_length])  # Encoding for the motion
        noise = torch.FloatTensor(1, audio_feat_seq_length, self.aux_latent).normal_(0, 0.33).to(self.device)
        z_id, skips = self.encoder_id(frame, retain_intermediate=True)
        skip_connections = []
        for skip_variable in skips:
            skip_connections.append(self._broadcast_elements_(skip_variable, z.size()[1]))
        skip_connections.reverse()

        z_id = self._broadcast_elements_(z_id, z.size()[1])
        gen_video = self.generator(z, c=z_id, aux=noise, skip=skip_connections)

        returned_audio = ((2 ** 15) * speech.detach().cpu().numpy()).astype(np.int16)
        gen_video = 125 * gen_video.squeeze().detach().cpu().numpy() + 125
        return gen_video, returned_audio
