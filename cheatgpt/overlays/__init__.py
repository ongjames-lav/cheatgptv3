"""
CheatGPT Overlays Package
Video overlay system for suspicious event visualization and recording
"""

from .hotspot_overlay import HotspotOverlay, EventDatabase, EngineOverlayIntegration
from .overlay_recorder import OverlayVideoRecorder, DetectionRecorderIntegration

__all__ = [
    'HotspotOverlay',
    'EventDatabase', 
    'EngineOverlayIntegration',
    'OverlayVideoRecorder',
    'DetectionRecorderIntegration'
]
