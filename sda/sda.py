from torchvision import transforms
import torch
import torchaudio
import encoder_image
import img_generator
import rnn_audio
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


class VideoAnimator():
    def __init__(self, model_path, gpu=-1):

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

        self.encoder = rnn_audio.RNN(self.audio_feat_len, self.aud_enc_dim, self.rnn_gen_dim,
                                     self.audio_rate, init_kernel=0.005, init_stride=0.001)
        self.encoder.load_state_dict(model_dict['encoder'])

        self.encoder_id = encoder_image.Encoder(self.id_enc_dim, self.img_size)
        self.encoder_id.load_state_dict(model_dict['encoder_id'])

        skip_channels = list(self.encoder_id.channels)
        skip_channels.reverse()

        self.generator = img_generator.Generator(self.img_size, self.rnn_gen_dim, condition_size=self.id_enc_dim,
                                                 num_gen_channels=self.encoder_id.channels[-1],
                                                 skip_channels=skip_channels, aux_size=self.aux_latent,
                                                 sequential_noise=self.sequential_noise)

        self.generator.to(self.device)
        self.generator.load_state_dict(model_dict['generator'])

    def save_video(self, video, audio, path):
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

            out = ffmpeg.output(in1['v'], in2['a'], path, loglevel="panic")
            out.run()
    def _cut_sequence_(self, seq, snip_length, cutting_stride, pad_samples):
        if cutting_stride is None:
            cutting_stride = snip_length

        pad_left = torch.zeros(pad_samples // 2, 1)
        pad_right = torch.zeros(pad_samples - pad_samples // 2, 1)

        seq = torch.cat((pad_left, seq), 0)
        seq = torch.cat((seq, pad_right), 0)

        stacked = seq.narrow(0, 0, snip_length).unsqueeze(0)
        iterations = (seq.size()[0] - snip_length) // cutting_stride + 1
        for i in range(1, iterations):
            stacked = torch.cat((stacked, seq.narrow(0, i * cutting_stride, snip_length).unsqueeze(0)))
        return stacked

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
        else:
            if fs is None:
                raise AttributeError("Audio provided without specifying the rate. Specify rate or use audio file!")
            seq_length = audio.shape[0]
            max_value = np.iinfo(img.dtype).max
            speech = torch.from_numpy(2 * signal.resample(audio, seq_length * self.audio_rate / fs) / max_value).float()

        frame = self.img_transform(frame).to(self.device)
        speech = speech.to(self.device)

        cutting_stride = int(self.audio_rate / self.video_rate)
        audio_seq_padding = self.audio_feat_samples - cutting_stride

        # Create new sequences of the audio windows
        audio_feat_seq = self._cut_sequence_(speech, self.audio_feat_samples, cutting_stride, audio_seq_padding)
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
