import numpy as np

from video2world.scale_alignment import fit_scale_shift


def test_fit_scale_shift_recovers_linear_depth_alignment_with_outlier():
    predicted = np.array([1.0, 2.0, 3.0, 4.0, 50.0])
    sparse_depth = np.array([3.0, 5.0, 7.0, 9.0, -20.0])

    result = fit_scale_shift(predicted, sparse_depth, trim_quantile=0.2)

    assert result.success
    assert result.num_inliers == 3
    assert abs(result.scale - 2.0) < 1e-6
    assert abs(result.shift - 1.0) < 1e-6


def test_fit_scale_shift_fails_when_too_few_valid_points():
    result = fit_scale_shift(np.array([np.nan, 2.0]), np.array([1.0, np.nan]))

    assert not result.success
    assert result.num_inliers == 0
