from __future__ import annotations

import pytest

from yifinance_rag.config import ConfigError, load_settings
from yifinance_rag.data.dataset import Dataset

# Resolved at import (collection), before the autouse fixture points HOME at a temp dir:
# FLOWPAY_DATASET_ROOT usually lives in the real ~/.yfr/.env.
try:
    _DATASET: Dataset | None = Dataset.from_settings(load_settings())
    _WHY = ""
except ConfigError as e:
    _DATASET, _WHY = None, str(e)


@pytest.fixture(scope="session")
def ds() -> Dataset:
    if _DATASET is None:
        pytest.fail(f"the dataset tests need the real dataset: {_WHY}")
    return _DATASET
