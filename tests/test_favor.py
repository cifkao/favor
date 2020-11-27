import numpy as np
import pytest
import torch

from favor import FAVOR


SOFTMAX_KERNEL = dict(
    h=lambda x: torch.exp(-torch.sum(x ** 2, dim=-1, keepdim=True) / 2) / 2 ** .5,
    f=[torch.exp, lambda u: torch.exp(-u)]
)


class TestFAVOR:

    def setup_method(self, method):
        torch.random.manual_seed(0)

    def test_softmax(self):
        d = 256
        att = FAVOR(d, m=256, **SOFTMAX_KERNEL)

        k = torch.randn(d, 10)[None, ...] / 2
        v = torch.arange(10.)[None, None, :]
        q = k

        result = att(k, v, q)
        expected = v @ softmax_attention(k, q)
        print(result)
        print(expected)
        np.testing.assert_allclose(result, expected)


def softmax_attention(keys, queries):
    logits = keys.transpose(-1, -2) @ queries / keys.shape[-2] ** .5
    return torch.softmax(logits, dim=-2)
