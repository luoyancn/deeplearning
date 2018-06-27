# -*- coding:utf-8 -*-

import numpy as np

SEED = 2

# 构造数据
def generate_dataset():
    rdm = np.random.RandomState(SEED)
    X = rdm.randn(300, 2)
    # 构造标准答案
    Y_ = [int(x0 * x0 + x1*x1) < 2 for (x0, x1) in X]
    Y_c = [['red' if y else 'blue'] for y in Y_]
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)

    return X, Y_, Y_c