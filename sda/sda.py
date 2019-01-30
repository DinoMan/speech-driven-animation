from torchvision import transforms
import torch
import torchaudio
import encoder_audio
import encoder_image
import img_generator
import rnn_audio

import
from PIL import Image


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
            transforms.Scale((self.img_size[0], self.img_size[1])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.audio_transform = torchaudio.transforms.Scale()

        self.encoder = rnn_audio.RNN(self.audio_feat_len, self.aud_enc_dim, self.rnn_gen_dim,
                                     self.audio_rate, init_kernel=0.005, init_stride=0.001)
        self.encoder_id = encoder_image.Encoder(self.id_enc_dim, self.img_size)

        skip_channels = self.encoder_id.channels
        skip_channels.reverse()
        self.generator = img_generator.Generator(self.img_size, self.rnn_gen_dim, condition_size=self.id_enc_dim,
                                                 num_gen_channels=self.encoder_id.channels[-1],
                                                 skip_channels=skip_channels, aux_size=self.aux_latent,
                                                 sequential_noise=self.sequential_noise)

        self.generator.to(self.device)

    def _cut_sequence_(seq, snip_length, cutting_stride, pad_samples):
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

    def _broadcast_elements_(batch, repeat_no):
        total_tensors = []
        for i in range(0, batch.size()[0]):
            total_tensors += [torch.stack(repeat_no * [batch[i]])]

        return torch.stack(total_tensors)

    def __call__(self, img, audio):
        if isinstance(img, str):  # if we have a path then grab the image
            frame = Image.open(img)
        else:
            frame = img

        if isinstance(audio, str):  # if we have a path then grab the image
            speech, audio_rate = torchaudio.load(audio)
        else:
            speech = audio

        frame = self.img_transform(frame).to(self.device)
        speech = self.audio_transform(audio).to(self.device)

        cutting_stride = int(self.audio_rate / self.video_rate)
        audio_seq_padding = self.audio_feat_samples - cutting_stride

        # Create new sequences of the audio windows
        audio_feat_seq = self._cut_sequence_(speech, self.audio_feat_samples, cutting_stride, audio_seq_padding)
        audio_feat_length = audio_feat_seq.size(1)

        z = self.encoder(audio_feat_seq, [audio_feat_length])  # Encoding for the motion
        noise = torch.FloatTensor(len(audio), audio_feat_length, self.aux_latent).normal_(0, 0.33).to(self.device)

        z_id, skips = self.encoder_id(frame, retain_intermediate=self.use_unet)
        for skip_variable in skips:
            skip_connections.append(self._broadcast_elements_(skip_variable, z.size()[1]))
        skip_connections.reverse()

        z_id = self._broadcast_elements_(z_id, z.size()[1])
        gen_video = self.generator(z, c=z_id, aux=noise, skip=skip_connections)
        return gen_video
