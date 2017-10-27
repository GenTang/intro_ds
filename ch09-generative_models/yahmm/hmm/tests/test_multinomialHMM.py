# -*- coding: UTF-8 -*-
import numpy as np
from hmm.multinomialHMM import MultinomialHMM
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_mutlnomialhmm():
    """
    """
    mh = MultinomialHMM(alpha=1)
    Y = [0, 1, 1, 1]
    X = [[1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1]]
    mh.fit(X, Y, [4])
    assert_array_almost_equal(np.exp(mh.initProb_), [2./3, 1./3])
    assert_array_almost_equal(np.exp(mh.transProb_),
        [[1./3, 2./3], [1./4, 3./4]])
    assert_array_almost_equal(np.exp(mh.emitProb_),
        [[2./4, 1./4, 1./4], [1./7, 3./7, 3./7]])
    mh = MultinomialHMM(alpha=0.1)
    mh.fit(X, Y, [2, 2])
    assert_array_almost_equal(np.exp(mh.initProb_), [1.1/2.2, 1.1/2.2])
    assert_array_almost_equal(np.exp(mh.transProb_),
        [[.1/1.2, 1.1/1.2], [.1/1.2, 1.1/1.2]])
    assert_array_almost_equal(np.exp(mh.emitProb_),
        [[1.1/1.3, .1/1.3, .1/1.3], [.1/4.3, 2.1/4.3, 2.1/4.3]])
    assert_array_almost_equal(mh.predict(X), mh.predict(X, lengths=[1, 3]))
