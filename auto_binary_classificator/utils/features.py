from typing import Union
import numpy as np
import pandas as pd


class Features:
    """

    """
    def __init__(self, data: Union[pd.DataFrame, np.array]):
        pass

    def transform(self, data):
        pass

    def _transform_text_to_category(self, data: Union[pd.Series, np.array]) -> np.array:
        """
        Converts text features to categorical.
        data: np.array
        """
        pass


Features()
