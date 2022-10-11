"""Transforms"""
import random
from typing import Tuple

import torch
import torchaudio
from pedalboard import Reverb
from torch import Tensor, nn


class GaussianNoise(nn.Module):
    """Gaussian Noise Transform."""

    def __init__(self, min_intensity: float = 0.0, max_intensity: float = 1.0):
        super().__init__()
        self.intensity_dist = torch.distributions.uniform.Uniform(
            min_intensity, max_intensity
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        intensity = self.intensity_dist.sample().to(x.device)
        noise = torch.randn_like(x) * intensity
        x += noise
        return x


class FilterTransform(nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        freq_ceil=12000,
        freq_floor=0,
        gain_ceil=20,
        gain_floor=-20,
        q=0.707,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.freq_ceil = freq_ceil
        self.freq_floor = freq_floor
        self.gain_ceil = gain_ceil
        self.gain_floor = gain_floor
        self.q = q

    def get_gain(self):
        return (self.gain_floor - self.gain_ceil) * random.random() + self.gain_ceil

    def get_center_freq(self):
        return (self.freq_floor - self.freq_ceil) * random.random() + self.freq_ceil

    def forward(self, x):
        gain = self.get_gain()
        center_freq = self.get_center_freq()

        x = torchaudio.functional.equalizer_biquad(
            x,
            sample_rate=self.sample_rate,
            center_freq=center_freq,
            gain=gain,
            Q=self.q,
        )
        return x


class ClipTransform(nn.Module):
    def __init__(self, clip_ceil=1, clip_floor=0.5):
        super().__init__()
        self.clip_ceil = clip_ceil
        self.clip_floor = clip_floor

    def get_clip(self):
        return (self.clip_floor - self.clip_ceil) * random.random() + self.clip_ceil

    def forward(self, x):
        clip_level = self.get_clip()
        x[torch.abs(x) > clip_level] = clip_level
        return x


class BreakTransform(nn.Module):
    def __init__(
        self, sample_rate=24000, break_duration=0.01, break_ceil=50, break_floor=10
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.break_segment = sample_rate * break_duration
        self.break_ceil = break_ceil
        self.break_floor = break_floor

    def get_mask(self, x):
        break_count = (
            self.break_floor - self.break_ceil
        ) * random.random() + self.break_ceil
        break_duration = break_count * self.break_segment
        mask = torch.ones(x.size())
        break_start = int(x.size(0) * random.random())
        break_end = int(min(x.size(0), break_start + break_duration))
        mask[break_start:break_end] = 0
        return mask

    def forward(self, x):
        break_mask = self.get_mask(x)
        x = x * break_mask
        return x


class MixTransform(nn.Module):
    def __init__(self, snr_ceil=30, snr_floor=-5):
        super().__init__()
        self.snr_ceil = snr_ceil
        self.snr_floor = snr_floor

    def get_snr(self, n):
        return (self.snr_floor - self.snr_ceil) * torch.rand([n]) + self.snr_ceil

    def forward(self, speech, noise):
        samples = speech.size(0)
        snr = self.get_snr(samples)
        noise = noise * torch.norm(speech) / torch.norm(noise)
        scalar = torch.pow(10.0, (0.05 * snr)).reshape([speech.size(0), 1])
        noise = torch.div(noise, scalar)
        mix = speech + noise
        return mix


class ReverbTransform(nn.Module):
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.sample_rate = sample_rate
        self.reverb = Reverb()

    @torch.no_grad()
    def forward(self, x):
        self.reverb.room_size = random.random()
        reverbed = self.reverb.process(x.numpy(), self.sample_rate)
        return reverbed


class SpecTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_hp = torch.tensor([-1.99599, 0.99600])
        self.b_hp = torch.tensor([-2, 1])

    def _uni_rand(self):
        return torch.rand(1) - 0.5

    def _rand_resp(self):
        a1 = 0.75 * self._uni_rand()
        a2 = 0.75 * self._uni_rand()
        b1 = 0.75 * self._uni_rand()
        b2 = 0.75 * self._uni_rand()
        return a1, a2, b1, b2

    def forward(self, x):
        a1, a2, b1, b2 = self._rand_resp()
        x = torchaudio.functional.biquad(
            x, 1, self.b_hp[0], self.b_hp[1], 1, self.a_hp[0], self.a_hp[1]
        )
        x = torchaudio.functional.biquad(x, 1, b1, b2, 1, a1, a2)
        return x


class VolTransform(nn.Module):
    def __init__(self, sample_rate=24000, segment_len=0.5, vol_ceil=10, vol_floor=-10):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_len = segment_len
        self.segment_samples = int(self.sample_rate * self.segment_len)
        self.vol_ceil = vol_ceil
        self.vol_floor = vol_floor

    def get_vol(self, sample_length):
        segments = sample_length / (self.segment_len * self.sample_rate)
        step_db = torch.arange(
            self.vol_ceil, self.vol_floor, (self.vol_floor - self.vol_ceil) / segments
        )
        return step_db

    def apply_gain(self, segments, db):
        gain = torch.pow(10.0, (0.05 * db))
        segments = segments * gain
        return segments

    def forward(self, x):
        step_db = self.get_vol(x.size(0))
        for i in range(step_db.size(0)):
            start = i * self.segment_samples
            end = min((i + 1) * self.segment_samples, x.size(0))
            x[:, start:end] = self.apply_gain(x[:, start:end], step_db[i])

        return x


class RandomTransform(nn.Module):
    """Randomly apply list of transforms."""

    def __init__(
        self,
        transforms: Tuple[nn.Module] = (
            GaussianNoise(),
            FilterTransform(),
            ClipTransform(),
            BreakTransform(),
            MixTransform(),
            ReverbTransform(),
            SpecTransform(),
            VolTransform(),
        ),
        probability=0.5,
    ):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass."""
        for t in self.transforms:
            if random.random() < self.probability:
                x = t(x)
        return x
