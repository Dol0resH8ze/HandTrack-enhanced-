import numpy as np
import sys
sys.path.insert(0, "src")
import features as F

def test_feature_vector_length():
    # make sure extract() always returns exactly 97 features
    dummy = np.random.rand(21, 3)
    result = F.extract(dummy)
    assert result.shape == (97,)

def test_normalization():
    # wrist should be at origin after normalization
    dummy = np.random.rand(21, 3)
    result = F.extract(dummy)
    # normalized coords start at index 0, wrist is first 3 values (x,y,z)
    assert abs(result[0]) < 1e-5
    assert abs(result[1]) < 1e-5
    assert abs(result[2]) < 1e-5
