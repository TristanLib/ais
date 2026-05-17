"""
Ship Trajectory Prediction and Collision Avoidance System

A comprehensive deep learning framework for predicting ship trajectories
and optimizing collision avoidance maneuvers based on AIS data.
"""

__version__ = "1.0.0"
__author__ = "Ship Prediction Research Team"

# Core modules
from . import dataio
from . import features
from . import models
from . import risk
from . import avoidance
from . import eval
from . import viz

__all__ = [
    'dataio',
    'features',
    'models',
    'risk',
    'avoidance',
    'eval',
    'viz'
]
