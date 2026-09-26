import numpy as np
import pandas as pd

from gpr_ml.core import (
    indices_from_split,
    load_split,
    save_split,
)


def test_split_helpers_round_trip_with_dataframe_rows(tmp_path):
    train = pd.DataFrame(
        {"reaction_id": ["A", "B"]}
    )
    test = pd.DataFrame(
        {"reaction_id": ["C"]}
    )
    path = tmp_path / "split.json"

    save_split(train, test, path)
    train_ids, test_ids = load_split(path)

    frame = pd.DataFrame(
        {
            "reaction_id": ["A", "A", "B", "C"],
            "label": [1, 0, 1, 1],
        }
    )
    train_idx, test_idx = indices_from_split(
        frame,
        train_ids,
        test_ids,
    )

    np.testing.assert_array_equal(
        train_idx,
        np.array([0, 1, 2]),
    )
    np.testing.assert_array_equal(
        test_idx,
        np.array([3]),
    )
