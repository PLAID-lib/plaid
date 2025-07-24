import os

import joblib
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


class Test_Blocks:
    def test_clone(self, all_blocks):
        for block in all_blocks:
            clone(block)

    def test_save_load(self, all_blocks, tmp_path):
        for block in all_blocks:
            joblib.dump(block, os.path.join(tmp_path, "block_state.pkl"))
            loaded_block = joblib.load(os.path.join(tmp_path, "block_state.pkl"))
            with pytest.raises(NotFittedError):
                check_is_fitted(loaded_block)

    def test_get_set_params(self, all_blocks):
        for block in all_blocks:
            param_name = next(iter(block.get_params()))
            block.set_params(**{param_name: 0.0})
