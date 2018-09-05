# -*- coding: UTF-8 -*-
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from hmm.utils.viterbi import viterbi


def test_viterbi():
    """
    此测试用例来源于维基百科
    https://en.wikipedia.org/wiki/Viterbi_algorithm#Example
    """
    init = np.log([.6, .4])
    trans = np.log([[.7, .3],
                    [.4, .6]])
    emit = np.log([[.5, .4, .1],
                   [.1, .3, .6]])
    obs = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    score, path = viterbi(obs, init, trans, emit)
    wiki_score = [[0.3, 0.04], [0.084, 0.027], [0.00588, 0.01512]]
    assert_array_equal(path, [0, 0, 1])
    assert_array_almost_equal(np.exp(score), wiki_score)
