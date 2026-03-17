"""graphids.core — Domain layer: models, preprocessing, data loading.

Public API (import from here, not from submodules):
    from graphids.core import load_dataset, load_test_scenarios
    from graphids.core import get_model, register_model
    from graphids.core import process_dataset, CollatedGraphDataset
"""

__version__ = "1.0.0"

from graphids.core.data import load_dataset, load_test_scenarios
from graphids.core.models.registry import get as get_model
from graphids.core.models.registry import register as register_model
from graphids.core.preprocessing import process_dataset
from graphids.core.preprocessing.dataset import CollatedGraphDataset
