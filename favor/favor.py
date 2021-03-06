import torch
from torch import nn
import torch.nn.functional as F
from warnings import warn
import itertools
import math
import numpy as np


class FAVOR(nn.Module):
    """Fast Attention Via positive Orthogonal Random features"""
    def __init__(
        self,
        key_dim,
        orthonormal=True,
        multihead=True,
        causal=False,
        m=128,
        redraw=True,
        h=lambda x: 1.,
        f=[F.relu,],
        randomizer=torch.randn,
    ):
        super(FAVOR, self).__init__()
        self.key_dim = key_dim

        self.orthonormal=orthonormal
        self.multihead = multihead
        self.causal = causal
        self.redraw=redraw
        self.m = m
        self.h = h
        self.f = f
        self.randomizer=randomizer

        self._features = None
        self.register_buffer('phi_scale', torch.tensor(1./ math.sqrt(m)))


    def features(self):
        if self._features is None or self.redraw:
            self._features = self.randomizer(
                (self.m, self.key_dim),
                device=self.phi_scale.device,
                dtype=self.phi_scale.dtype
            )
            if self.orthonormal:
                self._features = torch.qr(
                    self._features.double())[0].to(self.phi_scale.dtype)
        return self._features


    def forward(self, keys, values, queries):
        """
        keys: (batch, heads, keys_dimension, *keys_locations)
        values: (batch, heads, values_dimension, *keys_locations)
        queries: (batch, heads, keys_dimension, *queries_locations)

        If multihead is False, the heads dimension is to be omitted.
        """
        if self.multihead:
            # hiding the heads dimension in the batch dimension
            num_heads = keys.shape[1]
            keys, values, queries = (x.view(-1, *x.shape[2:]) for x in (keys, values, queries))

        # flattening everything
        keys_locations = keys.shape[2:]
        queries_locations = queries.shape[2:]
        keys, values, queries = (x.view(*x.shape[:2], -1) for x in (keys, values, queries))

        if self.causal and keys_locations != queries_locations:
            raise ValueError(
                'Expected equal key and query locations with causal attention, got: '
                '{}, {}'.format(keys_locations, queries_locations))

        # getting to (batch, n, dim)
        keys, values, queries = (x.permute(0, 2, 1) for x in (keys, values, queries))

        # features are (m, key_dim). randomized here if necessary
        features = self.features()

        # getting the randomized features for keys and queries
        def phi(x):
            # x is (batch, n, key_dim)

            # projections are (batch, n, m)
            projections = torch.matmul(x, features.T)

            # (batch, n, r)
            return torch.cat(
                [f(projections) for f in self.f],
                dim = -1
            ) * self.h(x) * self.phi_scale

        # (batch, n_context, r)
        phi_k = phi(keys)
        # (batch, n, r)
        phi_q = phi(queries)

        if self.causal:
            # outer products of keys and values: (batch, n, r, dim)
            k_v_prod = torch.matmul(phi_k[:, :, :, None], values[:, :, None, :])

            out = torch.matmul(         # (batch, n, dim)
                phi_q[:, :, None, :],   # (batch, n, 1, r)
                k_v_prod.cumsum(dim=1)  # (batch, n, r, dim)
            ).squeeze(2)

            # normalization factors: (batch, n, 1)
            norm = torch.matmul(
                phi_q[:, :, None, :],           # (batch, n, 1, r)
                phi_k.cumsum(dim=1)[..., None]  # (batch, n, r, 1)
            ).squeeze(2)
        else:
            out = torch.matmul( # (batch, n, dim)
                phi_q,
                torch.matmul( # (batch, r, dim)
                    phi_k.permute(0, 2, 1), values
                )
            )

            # normalization factors: (batch, n, 1)
            norm = torch.matmul(
                phi_q,
                phi_k.sum(dim=1)[..., None] # (batch, r, 1)
            )

        out /= norm

        # restoring the desired shape
        out = out.permute(0, 2, 1)
        out = out.reshape(*out.shape[:2], *queries_locations)
        if self.multihead:
            out = out.view(-1, num_heads, *out.shape[1:])
        return out
