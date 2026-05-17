"""
Feature engineering modules for ship trajectory prediction.
"""

from .geo import GeoFeatureEngine, convert_to_utm, convert_to_local_meters

__all__ = [
    'GeoFeatureEngine',
    'convert_to_utm',
    'convert_to_local_meters'
]
