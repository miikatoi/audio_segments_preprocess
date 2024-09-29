from pydantic import BaseModel as _BaseModel
from pydantic import ConfigDict
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


class WaveformSample(BaseModel):
    waveform: np.ndarray
    sample_rate: int

    """
    NOTE Ensure backends are installed https://pytorch.org/audio/main/generated/torchaudio.load.html#torchaudio.load
    """

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
        waveform_data = waveform_data.numpy()

        return cls(waveform=waveform_data, sample_rate=sample_rate, **kwargs)

    def process(self, feature_extractor: "FeatureExtractor"):
        features = feature_extractor(self.waveform.squeeze(), sampling_rate=self.sample_rate, return_tensors="np")
        return SpectrumSample(spectrum=features["input_values"])


class SpectrumSample(BaseModel):
    spectrum: np.ndarray

    def plot(self):
        import librosa
        import matplotlib.pyplot as plt

        spec = self.spectrum.squeeze()
        db = librosa.amplitude_to_db(spec)
        plt.imshow(db, origin="lower", aspect="auto")
        plt.show()
        # if labels is not None:
        #     _labels = labels.copy()
        #     _labels[_labels == -100] = -1
        #     plt.plot(_labels)
        #     plt.show()
