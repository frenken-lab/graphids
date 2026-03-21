"""graphids.core — Domain layer: models, preprocessing, data loading.

Public API (import from here, not from submodules):
    from graphids.core import load_dataset, load_test_scenarios
    from graphids.core import get_model
    from graphids.core import CollatedGraphDataset
"""

__version__ = "1.0.0"

from graphids.core.models.registry import get as get_model
from graphids.core.preprocessing._cache import load_dataset, load_test_scenarios
from graphids.core.preprocessing._dataset import CollatedGraphDataset
