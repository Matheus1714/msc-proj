from typing import Tuple
import math

def integer_proportion(a: int, b: int) -> Tuple[int, int]:
  if a == 0 and b == 0:
      return (0, 0)
  if a == 0:
      return (0, 1)
  if b == 0:
      return (1, 0)
  mdc = math.gcd(a, b)
  return (a // mdc, b // mdc)
