import numpy as np
import math

def _inv_linear(val, low, high):
    return float(np.clip((val - low) / (high - low) * 2.0 - 1.0, -1.0, 1.0))

print("0 raw for compressor threshold [-60, 0]:", (0.0 + 1.0)*0.5*60 - 60)
print("0 raw for EQ filter type [0, 6]:", (0.0 + 1.0)*0.5*6)

