from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict, Field
import numpy as np
from typing import List, Optional


class BaseModel(_BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AudioSegmentLabel(BaseModel):
    start: float
    end: float
    label: str

    @property
    def class_index(self):
        raise NotImplementedError


class SegmentsAnnotation(BaseModel):
    segments: List[AudioSegmentLabel]


class WindowedAudioSample(BaseModel):
    """
    TODO
    implement methods to join multiple predictions together
    """

    original_sample: "AudioSample"
    start_idx: float
    end_idx: float


class AudioSample(BaseModel):
    sample_rate: int
    annotation: Optional[SegmentsAnnotation] = None

    def windowed(self, window_size: int):
        raise NotImplementedError


class WaveformSample(AudioSample):
    waveform: np.ndarray

    """
    NOTE Ensure backends are installed https://pytorch.org/audio/main/generated/torchaudio.load.html#torchaudio.load
    """

    class WindowedWaveformAudioSample(BaseModel):
        original_sample: "AudioSample"
        start_idx: float
        end_idx: float

    @classmethod
    def from_wav_file(cls, file_name: str, sample_rate: Optional[int] = None, **kwargs):
        import torchaudio

        waveform_data, _sample_rate = torchaudio.load(file_name, format="wav")
        if sample_rate is not None and _sample_rate != sample_rate:
            from torchaudio import transforms as T

            resample = T.Resample(_sample_rate, sample_rate)
            waveform_data = resample(waveform_data)
        else:
            sample_rate = _sample_rate
        waveform_data = waveform_data.numpy().squeeze()

        return cls(waveform=waveform_data, sample_rate=sample_rate, **kwargs)

    def process(self, feature_extractor: "FeatureExtractor"):
        features = feature_extractor(self.waveform, sampling_rate=self.sample_rate, return_tensors="np")
        return SpectrumSample(
            spectrum=features["input_values"].squeeze(),
            annotation=self.annotation,
            sample_rate=self.sample_rate,
        )

    def windowed(self, window_size: int):
        lenght = self.waveform.shape[0]
        return [
            self.WindowedWaveformAudioSample(
                original_sample=self,
                start_idx=start_idx,
                end_idx=min(start_idx + window_size, lenght),
            )
            for start_idx in range(0, lenght, window_size)
        ]


class SpectrumSample(AudioSample):
    spectrum: np.ndarray

    class WindowedSpectrumAudioSample(BaseModel):
        original_sample: "AudioSample"
        start_idx: float
        end_idx: float

    def plot(self):
        import librosa
        import matplotlib.pyplot as plt

        spec = self.spectrum.squeeze()
        db = librosa.amplitude_to_db(spec)
        plt.imshow(db, origin="lower", aspect="auto")
        plt.show()

    def windowed(self, window_size: int):
        lenght = self.spectrum.shape[0]  # need to ensure time dimension is first
        return [
            self.WindowedSpectrumAudioSample(
                original_sample=self,
                start_idx=start_idx,
                end_idx=min(start_idx + window_size, lenght),
            )
            for start_idx in range(0, lenght, window_size)
        ]
