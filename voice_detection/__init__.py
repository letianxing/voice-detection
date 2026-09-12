"""Cocktail-party acoustic frontend for robot HRI perception."""

from .aec import LinearEchoCanceller, PlaybackReferenceBuffer
from .frontend import CocktailFrontend
from .profiles import load_profiles
from .types import AcousticTrack, AudioFrame, FrontendOutput, MicrophoneProfile, SpeechTranscript

__all__ = [
    "AcousticTrack",
    "AudioFrame",
    "CocktailFrontend",
    "FrontendOutput",
    "MicrophoneProfile",
    "SpeechTranscript",
    "LinearEchoCanceller",
    "PlaybackReferenceBuffer",
    "load_profiles",
]
