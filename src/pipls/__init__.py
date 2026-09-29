"""Π-PLS multivariate regression."""

from .exceptions import PredictorRankSupportWarning
from .regression import PiPLSRegression
from .search import PiPLSSearchCV

__all__ = [
    "PiPLSRegression",
    "PiPLSSearchCV",
    "PredictorRankSupportWarning",
    "__version__",
]

__version__ = "0.1.0"
