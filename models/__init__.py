"""
模型模块
"""

from models.faces import FacesModel
from models.audio import AudioModel
from models.openface import OpenfaceModel
from models.fusion import FusionModel

__all__ = [
    'FacesModel',
    'AudioModel',
    'OpenfaceModel',
    'FusionModel'
]
