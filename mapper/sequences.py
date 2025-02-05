# -*- coding: utf-8 -*-
#
# Copyright 2018-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sequences to provide input to Keras

"""
__all__ = [
    "RelationalFullBatchNodeSequence",
]

import numpy as np
from tensorflow.keras.utils import Sequence


def _full_batch_array_and_reshape(array, propagate_none=False):
    """
    Args:
        array: an array-like object
        propagate_none: if True, return None when array is None
    Returns:
        array as a numpy array with an extra first dimension (batch dimension) equal to 1
    """
    # if it's ok, just short-circuit on None (e.g. for target arrays, that may or may not exist)
    if propagate_none and array is None:
        return None

    as_np = np.asanyarray(array)
    return np.reshape(as_np, (1,) + as_np.shape)


class RelationalFullBatchNodeSequence(Sequence):
    """
    Keras-compatible data generator for for node inference models on relational graphs
    that require full-batch training (e.g., RGCN).
    Use this class with the Keras methods :meth:`keras.Model.fit`,
        :meth:`keras.Model.evaluate`, and
        :meth:`keras.Model.predict`,

    This class uses either dense or sparse representations to send data to the models.

    This class should be created using the `.flow(...)` method of
    :class:`.RelationalFullBatchNodeGenerator`.

    Args:
        features (np.ndarray): An array of node features of size (N x F),
            where N is the number of nodes in the graph, F is the node feature size
        As (list of sparse matrices): A list of length R of adjacency matrices of the graph of size (N x N)
            where R is the number of relationships in the graph.
        targets (np.ndarray, optional): An optional array of node targets of size (N x C),
            where C is the target size (e.g., number of classes for one-hot class targets)
        indices (np.ndarray, optional): Array of indices to the feature and adjacency matrix
            of the targets. Required if targets is not None.
    """

    def __init__(self, features, As, use_sparse, targets=None, indices=None):

        if (targets is not None) and (len(indices) != len(targets)):
            raise ValueError(
                "When passed together targets and indices should be the same length."
            )

        self.use_sparse = use_sparse

        # Convert all adj matrices to dense and reshape to have batch dimension of 1
        if self.use_sparse:
            self.A_indices = [
                np.expand_dims(
                    np.hstack((A.row[:, None], A.col[:, None])).astype(np.int64), 0
                )
                for A in As
            ]
            self.A_values = [np.expand_dims(A.data, 0) for A in As]
            self.As = self.A_indices + self.A_values
        else:
            self.As = [np.expand_dims(A.todense(), 0) for A in As]

        # Make sure all inputs are numpy arrays, and have batch dimension of 1
        self.target_indices = _full_batch_array_and_reshape(indices)
        self.features = _full_batch_array_and_reshape(features)
        self.inputs = [self.features, self.target_indices] + self.As

        self.targets = _full_batch_array_and_reshape(targets, propagate_none=True)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.inputs, self.targets
