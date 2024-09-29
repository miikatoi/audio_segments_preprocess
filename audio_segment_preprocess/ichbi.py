from typing import Tuple, Dict, ClassVar

from audio_segment_preprocess.typing import AudioSegmentLabel, SegmentsAnnotation


class ICHBIAudioSegmentLabel(AudioSegmentLabel):
    ICBHI_LABEL_MAP: ClassVar[Dict[Tuple[str, str], str]] = {
        ("0", "0"): "normal",
        ("1", "0"): "crackle",
        ("0", "1"): "wheeze",
        ("1", "1"): "both",
    }

    @classmethod
    def from_ichbi_row(cls, row):
        start, end, crackle, wheeze = row.strip().split("\t")
        crackle, wheeze
        return AudioSegmentLabel(start=float(start), end=float(end), label=cls.ICBHI_LABEL_MAP[(crackle, wheeze)])


class ICHBISegmentsAnnotation(SegmentsAnnotation):

    @classmethod
    def from_ichbi_annotation(cls, annotation):
        return cls(segments=[ICHBIAudioSegmentLabel.from_ichbi_row(row) for row in annotation.split("\n")])
