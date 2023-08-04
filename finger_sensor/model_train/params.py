# -*- coding: utf-8 -*-

import attr

@attr.s(auto_attribs=True)
class Params:

    train_data_dir: str = "../data_set/train_data/train_data.pkl"
    test_data_dir: str = "../data_set/test_data/test_data.pkl"
    data_dir: str = "../data_set/data"

    batch_size = 100
    window_size: int = 32
    test_interval = 5  # test interval /epoch
    draw_key = 1
    epoch = 50
    lr = 2e-4
    d_model = 512
    d_hidden = 1024
    q = 8
    v = 8
    h = 8
    n = 8
    d_channel = 2
    d_output = 31
    dropout = 0.2