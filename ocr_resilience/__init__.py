from .engines import AVAILABLE_ENGINES, Detection
from .pipeline import OCR, OCRPipeline, OCRResult
from .quality import QualityReport, assess
from .router import RoutingDecision

__all__ = [
    "OCR",
    "OCRPipeline",
    "OCRResult",
    "Detection",
    "AVAILABLE_ENGINES",
    "QualityReport",
    "RoutingDecision",
    "assess",
]
__version__ = "0.5.0"
