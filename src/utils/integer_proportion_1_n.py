from typing import Tuple
from math import floor

def integer_proportion_1_n(a: int, b: int) -> Tuple[int, int]:
    return (1, floor(b / a))
