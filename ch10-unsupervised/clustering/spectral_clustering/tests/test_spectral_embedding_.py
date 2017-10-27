# -*- coding: UTF-8 -*-
import numpy as np
from numpy.testing import assert_array_almost_equal
from spectral_clustering.spectral_embedding_ import spectral_embedding


def assert_first_col_equal(maps):
    constant_vec = [1] * maps.shape[0]
    assert_array_almost_equal(maps[:, 0] / maps[0, 0], constant_vec)


def test_spectral_embedding():
    """
    根据spectral embedding的定义，第一列的数据是恒等的
    """
    adjacency = np.array([
        [0., 0.8, 0.9, 0.],
        [0.8, 0., 0., 0.],
        [0.9, 0., 0., 1.],
        [0., 0., 1., 0.]])
    maps = spectral_embedding(
        adjacency, n_components=2, drop_first=False, eigen_solver="arpack")
    assert_first_col_equal(maps)
    maps_1 = spectral_embedding(
        adjacency, n_components=2, drop_first=False, eigen_solver="lobpcg")
    assert_first_col_equal(maps_1)
