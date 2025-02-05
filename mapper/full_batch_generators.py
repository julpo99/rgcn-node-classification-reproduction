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
Mappers to provide input data for the graph models in layers.

"""
__all__ = [
    "RelationalFullBatchNodeGenerator",
]

import numpy as np
import scipy.sparse as sps

from core.graph import StellarGraph
from core.utils import is_real_iterable
from core.validation import comma_sep
from . import (
    Generator,
    RelationalFullBatchNodeSequence,
)


class RelationalFullBatchNodeGenerator(Generator):
    """
    A data generator for use with full-batch models on relational graphs e.g. RGCN.

    The supplied graph G should be a StellarGraph or StellarDiGraph object with node features.
    Use the :meth:`flow` method supplying the nodes and (optionally) targets
    to get an object that can be used as a Keras data generator.

    This generator will supply the features array and the adjacency matrix to a
    full-batch Keras graph ML model.  There is a choice to supply either a list of sparse
    adjacency matrices (the default) or a list of dense adjacency matrices, with the `sparse`
    argument.

    For these algorithms the adjacency matrices require preprocessing and the default option is to
    normalize each row of the adjacency matrix so that it sums to 1.
    For customization a transformation (callable) can be passed that
    operates on the node features and adjacency matrix.

    Example::

        G_generator = RelationalFullBatchNodeGenerator(G)
        train_data_gen = G_generator.flow(node_ids, node_targets)

        # Fetch the data from train_data_gen, and feed into a Keras model:
        # Alternatively, use the generator itself with model.fit:
        model.fit(train_gen, epochs=num_epochs, ...)

    .. seealso::

       Model using this generator: :class:`.RGCN`.

       Examples using this generator:

       - `node classification <https://stellargraph.readthedocs.io/en/stable/demos/node-classification/rgcn-node-classification.html>`__
       - `unsupervised representation learning with Deep Graph Infomax <https://stellargraph.readthedocs.io/en/stable/demos/embeddings/deep-graph-infomax-embeddings.html>`__

       Related generators:

       - :class:`.FullBatchNodeGenerator` for graphs with one edge type
       - :class:`.CorruptedGenerator` for unsupervised training with :class:`.DeepGraphInfomax`

    Args:
        G (StellarGraph): a machine-learning StellarGraph-type graph
        name (str): an optional name of the generator
        transform (callable): an optional function to apply on features and adjacency matrix
            the function takes ``(features, Aadj)`` as arguments.
        sparse (bool): If True (default) a list of sparse adjacency matrices is used,
            if False a list of dense adjacency matrices is used.
        weighted (bool, optional): if True, use the edge weights from ``G``; if False, treat the
            graph as unweighted.
    """

    def __init__(self, G, name=None, sparse=True, transform=None, weighted=False):

        if not isinstance(G, StellarGraph):
            raise TypeError("Graph must be a StellarGraph object.")

        self.graph = G
        self.name = name
        self.use_sparse = sparse
        self.multiplicity = 1

        # Check if the graph has features
        G.check_graph_for_ml()

        # extract node, feature, and edge type info from G
        node_types = list(G.node_types)
        if len(node_types) != 1:
            raise ValueError(
                f"G: expected one node type, found {comma_sep(sorted(node_types))}",
            )

        self.features = G.node_features(node_type=node_types[0])

        # create a list of adjacency matrices - one adj matrix for each edge type
        # an adjacency matrix is created for each edge type from all edges of that type
        self.As = []

        for edge_type in G.edge_types:
            # note that A is the transpose of the standard adjacency matrix
            # this is to aggregate features from incoming nodes
            A = G.to_adjacency_matrix(
                edge_type=edge_type, weighted=weighted
            ).transpose()

            if transform is None:
                # normalize here and replace zero row sums with 1
                # to avoid harmless divide by zero warnings
                d = sps.diags(
                    np.float_power(np.ravel(np.maximum(A.sum(axis=1), 1)), -1), 0
                )
                A = d.dot(A)

            else:
                self.features, A = transform(self.features, A)

            A = A.tocoo()
            self.As.append(A)

    def num_batch_dims(self):
        return 2

    def flow(self, node_ids, targets=None, use_ilocs=False):
        """
        Creates a generator/sequence object for training or evaluation
        with the supplied node ids and numeric targets.

        Args:
            node_ids: and iterable of node ids for the nodes of interest
                (e.g., training, validation, or test set nodes)
            targets: a 2D array of numeric node targets with shape ``(len(node_ids), target_size)``
            use_ilocs (bool): if True, node_ids are represented by ilocs,
                otherwise node_ids need to be transformed into ilocs
        Returns:
            A NodeSequence object to use with RGCN models
            in Keras methods :meth:`fit`, :meth:`evaluate`,
            and :meth:`predict`
        """

        if targets is not None:
            # Check targets is an iterable
            if not is_real_iterable(targets):
                raise TypeError("Targets must be an iterable or None")

            # Check targets correct shape
            if len(targets) != len(node_ids):
                raise TypeError("Targets must be the same length as node_ids")

        if use_ilocs:
            node_indices = np.asarray(node_ids)
        else:
            node_indices = self.graph.node_ids_to_ilocs(node_ids)

        return RelationalFullBatchNodeSequence(
            self.features, self.As, self.use_sparse, targets, node_indices
        )

    def default_corrupt_input_index_groups(self):
        return [[0]]