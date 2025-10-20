"""
Vogent Turn - Real-time turn detection for conversational AI.

A lightweight library for detecting conversation turn endpoints using
multimodal analysis of audio and text context.

Main exports:
- TurnDetector: Turn detection inference class
"""

from .inference import TurnDetector

__version__ = "0.1.0"
__all__ = ["TurnDetector"]