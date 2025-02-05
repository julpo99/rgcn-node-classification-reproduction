import tensorflow as tf
from tensorflow.keras import activations, initializers, constraints, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Dropout, Input, Layer

from mapper.full_batch_generators import RelationalFullBatchNodeGenerator
from .misc import SqueezedSparseConversion, deprecated_model_function, GatherIndices


class RelationalGraphConvolution(Layer):
    """
    A Keras layer implementing the Relational Graph Convolution Network (RGCN) mechanism.

    Reference:
    Thomas N. Kipf & Michael Schlichtkrull (2017),
    "Modeling Relational Data with Graph Convolutional Networks"
    https://arxiv.org/pdf/1703.06103.pdf

    This layer expects:
      - A batch dimension of 1 (full-batch setup),
      - Node features as the first input,
      - Followed by one adjacency matrix per relationship.

    Args:
        units (int): Dimensionality of each node's transformed features.
        num_relationships (int): Number of distinct relationship types.
        num_bases (int): Number of basis matrices to factorize the relational parameters
            (use 0 or negative to disable basis decomposition).
        activation (str or callable): Activation function (e.g. 'relu', tf.nn.relu, ...).
        use_bias (bool): If True, includes a trainable bias term.
        final_layer (bool): Deprecated; use "tf.gather" or "GatherIndices" externally if needed.
        input_dim (int): (Optional) Size of the last axis of the node features if known in advance.
        kernel_initializer: Initializer for self and relational kernels (if no basis decomposition).
        kernel_regularizer: Regularizer for self and relational kernels.
        kernel_constraint: Constraint for self and relational kernels.
        bias_initializer: Initializer for the bias term.
        bias_regularizer: Regularizer for the bias term.
        bias_constraint: Constraint for the bias term.
        basis_initializer: Initializer for basis matrices if num_bases > 0.
        basis_regularizer: Regularizer for basis matrices.
        basis_constraint: Constraint for basis matrices.
        coefficient_initializer: Initializer for basis coefficients if num_bases > 0.
        coefficient_regularizer: Regularizer for basis coefficients.
        coefficient_constraint: Constraint for basis coefficients.
        kwargs: Standard Keras Layer keyword arguments.
    """

    def __init__(
            self,
            units,
            num_relationships,
            num_bases=0,
            activation=None,
            use_bias=True,
            final_layer=None,
            input_dim=None,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            kernel_constraint=None,
            bias_initializer="zeros",
            bias_regularizer=None,
            bias_constraint=None,
            basis_initializer="glorot_uniform",
            basis_regularizer=None,
            basis_constraint=None,
            coefficient_initializer="glorot_uniform",
            coefficient_regularizer=None,
            coefficient_constraint=None,
            **kwargs
    ):
        # If input_dim is known, pass it via input_shape to the parent class
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        # Initialize parent class
        super(RelationalGraphConvolution, self).__init__(**kwargs)

        # Validate arguments
        if not isinstance(units, int):
            raise TypeError("'units' must be an integer.")
        if units <= 0:
            raise ValueError("'units' must be a positive integer.")
        if not isinstance(num_relationships, int):
            raise TypeError("'num_relationships' must be an integer.")
        if num_relationships <= 0:
            raise ValueError("'num_relationships' must be > 0.")
        if not isinstance(num_bases, int):
            raise TypeError("'num_bases' must be an integer.")

        # Assign layer parameters
        self.units = units
        self.num_relationships = num_relationships
        self.num_bases = num_bases
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        # Manage user-provided or default config for kernel, bias, basis, coefficient
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.basis_initializer = initializers.get(basis_initializer)
        self.basis_regularizer = regularizers.get(basis_regularizer)
        self.basis_constraint = constraints.get(basis_constraint)

        self.coefficient_initializer = initializers.get(coefficient_initializer)
        self.coefficient_regularizer = regularizers.get(coefficient_regularizer)
        self.coefficient_constraint = constraints.get(coefficient_constraint)

        if final_layer is not None:
            raise ValueError(
                "The 'final_layer' argument is no longer supported. "
                "Use 'tf.gather' or 'GatherIndices' in your graph if needed."
            )

        # Call super constructor again (as done in the original)
        super(RelationalGraphConvolution, self).__init__(**kwargs)

    def get_config(self):
        """
        Returns a Python dictionary containing the layer configuration,
        enabling Keras to serialize/deserialize this layer.
        """
        cfg = super(RelationalGraphConvolution, self).get_config()
        cfg.update(
            {
                "units": self.units,
                "use_bias": self.use_bias,
                "activation": activations.serialize(self.activation),
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "basis_initializer": initializers.serialize(self.basis_initializer),
                "coefficient_initializer": initializers.serialize(self.coefficient_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "basis_regularizer": regularizers.serialize(self.basis_regularizer),
                "coefficient_regularizer": regularizers.serialize(self.coefficient_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "basis_constraint": constraints.serialize(self.basis_constraint),
                "coefficient_constraint": constraints.serialize(self.coefficient_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
                "num_relationships": self.num_relationships,
                "num_bases": self.num_bases,
            }
        )
        return cfg

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape from the provided input shapes.

        Args:
            input_shapes (list/tuple): The shapes of the inputs
                (the first for node features, then adjacency matrices).

        Returns:
            tuple: The shape of the output tensor.
        """
        features_shape = input_shapes[0]
        # We expect a batch dimension of size 1 for full-batch
        batch_dim = features_shape[0]
        n_nodes = features_shape[1]
        return (batch_dim, n_nodes, self.units)

    def build(self, input_shapes):
        """
        Constructs the layer's weights given the input shapes.

        Args:
            input_shapes (list/tuple): Shapes of the inputs
                (node features and adjacency matrices).
        """
        feat_shape = input_shapes[0]
        input_dim = int(feat_shape[-1])

        # If basis decomposition is enabled
        if self.num_bases > 0:
            # Basis matrices: shape (input_dim, units, num_bases)
            self.bases = self.add_weight(
                shape=(input_dim, self.units, self.num_bases),
                initializer=self.basis_initializer,
                name="bases",
                regularizer=self.basis_regularizer,
                constraint=self.basis_constraint,
            )

            # Coefficients for each relationship (length = num_relationships)
            self.coefficients = []
            for r in range(self.num_relationships):
                coeffs = self.add_weight(
                    shape=(self.num_bases,),
                    initializer=self.coefficient_initializer,
                    name="coeff",
                    regularizer=self.coefficient_regularizer,
                    constraint=self.coefficient_constraint,
                )
                self.coefficients.append(coeffs)

            # We'll compute relational_kernels dynamically in call()
            self.relational_kernels = None

        else:
            # If no basis decomposition is used, we have distinct kernels for each relation
            self.bases = None
            self.coefficients = None
            self.relational_kernels = []
            for r in range(self.num_relationships):
                w_rel = self.add_weight(
                    shape=(input_dim, self.units),
                    initializer=self.kernel_initializer,
                    name="relational_kernels",
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                )
                self.relational_kernels.append(w_rel)

        # Self-connection kernel
        self.self_kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="self_kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        # Optional bias
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        """
        Forward-pass: applies the RGCN transformation.

        Args:
            inputs (list):
                - The first element is a node feature tensor of shape (1, N, F).
                - The following elements are adjacency matrices (one for each relationship),
                  each shape (N, N).

        Returns:
            tf.Tensor of shape (1, N, units) representing the updated node embeddings.
        """
        # Unpack inputs
        features, *As = inputs

        # Expecting the batch dimension to be 1
        shape_nf = K.int_shape(features)
        if shape_nf[0] != 1:
            raise ValueError("This layer requires a batch dimension of exactly 1.")

        # Remove the batch dimension to simplify operations
        features = K.squeeze(features, 0)

        # Contribution from self-kernel
        out_transformed = K.dot(features, self.self_kernel)

        # If bases are used, compute relational weights from them
        if self.relational_kernels is None:
            # Combine basis matrices with each relationship's coefficients
            # => 1 weight matrix per relationship
            updated_rel_kernels = [
                tf.einsum("ijk,k->ij", self.bases, cff) for cff in self.coefficients
            ]
        else:
            updated_rel_kernels = self.relational_kernels

        # Accumulate neighbor contributions from each relationship
        for i in range(self.num_relationships):
            neighbor_feats = K.dot(As[i], features)
            out_transformed += K.dot(neighbor_feats, updated_rel_kernels[i])

        # Bias if applicable
        if self.bias is not None:
            out_transformed = out_transformed + self.bias

        # Activation
        out_activated = self.activation(out_transformed)

        # Reintroduce the batch dimension
        out_final = K.expand_dims(out_activated, 0)

        return out_final


class RGCN:
    """
    A relational GCN model, stacking multiple RelationalGraphConvolution layers
    to process graphs with multiple relationship (edge) types, as per
    "Modeling Relational Data with Graph Convolutional Networks"
    (Kipf & Schlichtkrull, 2017).

    Overview:
      - Takes a list of layer sizes (int) to define the hidden dimensions.
      - Has matching activation functions per layer.
      - Uses a `RelationalFullBatchNodeGenerator` to produce input tensors,
        including adjacency representations (sparse or dense).

    Notes:
      - The layer requires the batch dimension to be exactly 1 (full-batch).
      - Node ordering is consistent with the adjacency matrix; the final layer
        uses a gather operation to extract the requested node outputs.
      - For usage, see the `RelationalFullBatchNodeGenerator` documentation
        and the examples in the StellarGraph library.

    Example:
        .. code-block:: python

            gen = RelationalFullBatchNodeGenerator(G)
            rgcn_model = RGCN(
                layer_sizes=[32, 4],
                activations=["elu","softmax"],
                num_bases=10,
                generator=gen,
                dropout=0.5
            )
            x_inp, out_prediction = rgcn_model.in_out_tensors()

    Args:
        layer_sizes (list of int): Dimensions of each RGCN layer in the model.
        generator (RelationalFullBatchNodeGenerator): Data source; must be a
            `RelationalFullBatchNodeGenerator`.
        bias (bool): Whether to include a bias term in the RGCN layers.
        num_bases (int): Number of basis matrices to use in decomposition; 0 or negative means none.
        dropout (float): Dropout rate applied before each RGCN layer.
        activations (list of str or callable): Activation functions per layer. Defaults to 'relu' if None.
        kernel_initializer: Initializer for weights in each RGCN layer.
        kernel_regularizer: Regularizer for weights in each RGCN layer.
        kernel_constraint: Constraint for weights in each RGCN layer.
        bias_initializer: Initializer for bias terms in each RGCN layer.
        bias_regularizer: Regularizer for bias terms.
        bias_constraint: Constraint for bias terms.
    """

    def __init__(
            self,
            layer_sizes,
            generator,
            bias=True,
            num_bases=0,
            dropout=0.0,
            activations=None,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            kernel_constraint=None,
            bias_initializer="zeros",
            bias_regularizer=None,
            bias_constraint=None,
    ):
        # Validate the generator type
        if not isinstance(generator, RelationalFullBatchNodeGenerator):
            raise TypeError(
                "Expected generator to be a RelationalFullBatchNodeGenerator instance."
            )

        # Basic parameters
        self.layer_sizes = layer_sizes
        self.use_bias = bias
        self.num_bases = num_bases
        self.dropout_rate = dropout

        # Copy essential info from the generator
        self.multiplicity = generator.multiplicity
        self.n_nodes = generator.features.shape[0]
        self.num_features = generator.features.shape[1]
        self.num_relations = len(generator.As)
        self.use_sparse = generator.use_sparse

        # Number of layers to construct
        num_layers = len(layer_sizes)

        # Set or check activation functions
        if activations is None:
            activations = ["relu"] * num_layers
        elif len(activations) != num_layers:
            raise ValueError(
                "The number of activations must match the number of layers."
            )
        self.activations = activations

        # Build up the stack of RGCN layers + dropout
        self.model_layers = []
        for i in range(num_layers):
            self.model_layers.append(Dropout(self.dropout_rate))
            self.model_layers.append(
                RelationalGraphConvolution(
                    units=self.layer_sizes[i],
                    num_relationships=self.num_relations,
                    num_bases=self.num_bases,
                    activation=self.activations[i],
                    use_bias=self.use_bias,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_initializer=bias_initializer,
                    bias_regularizer=bias_regularizer,
                    bias_constraint=bias_constraint,
                )
            )

    def __call__(self, inputs):
        """
        Applies the RGCN stack to the given inputs.

        Expected inputs for a full-batch setup:
            1) Node features, shape=(1, N, F)
            2) Output node indices, shape=(1, Z)
            3) Then adjacency data: either (indices, values) pairs if sparse,
               or a dense adjacency matrix for each relationship.

        Args:
            inputs (list): The input tensors for the model.

        Returns:
            A tensor containing the outputs for the requested nodes.
        """
        x_in, out_indices, *As = inputs

        # Validate that the batch dimension is 1
        b_dim, n_nodes, _ = K.int_shape(x_in)
        if b_dim != 1:
            raise ValueError("Full-batch RGCN requires a batch dimension of exactly 1.")

        # For sparse adjacency
        if self.use_sparse:
            adj_indices_list = As[: self.num_relations]
            adj_values_list = As[self.num_relations:]

            # Convert each adjacency to a SqueezedSparseConversion
            final_As = []
            for idx, val in zip(adj_indices_list, adj_values_list):
                sc = SqueezedSparseConversion(
                    shape=(n_nodes, n_nodes), dtype=val.dtype
                )([idx, val])
                final_As.append(sc)
        else:
            # For dense adjacency, just remove the batch dimension
            final_As = [Lambda(lambda x: K.squeeze(x, 0))(A) for A in As]

        # Sequentially apply the layers
        h = x_in
        for layer in self.model_layers:
            # If it's an RGCN layer, pass adjacency as well
            if isinstance(layer, RelationalGraphConvolution):
                h = layer([h] + final_As)
            else:
                h = layer(h)

        # Gather the results for the nodes of interest
        out_tensor = GatherIndices(batch_dims=1)([h, out_indices])
        return out_tensor

    def _node_model(self):
        """
        Assembles the RGCN model for node-level tasks.

        Returns:
            (x_in, x_out):
            - x_in: list of input tensors (features, node indices, adjacency info),
            - x_out: the output tensor after applying the RGCN stack.
        """

        # Inputs for the node features and the output indices
        x_in_input = Input(batch_shape=(1, self.n_nodes, self.num_features))
        out_indices_input = Input(batch_shape=(1, None), dtype="int32")

        if self.use_sparse:
            # Sparse adjacency placeholders
            adj_indices_list = [
                Input(batch_shape=(1, None, 2), dtype="int64")
                for _ in range(self.num_relations)
            ]
            adj_values_list = [
                Input(batch_shape=(1, None)) for _ in range(self.num_relations)
            ]
            adjacency_placeholders = adj_indices_list + adj_values_list
        else:
            # Dense adjacency placeholders
            adjacency_placeholders = [
                Input(batch_shape=(1, self.n_nodes, self.n_nodes))
                for _ in range(self.num_relations)
            ]

        # Combine all input tensors
        x_input_tensors = [x_in_input, out_indices_input] + adjacency_placeholders
        # Pass them through our model (via __call__)
        x_output_tensor = self(x_input_tensors)

        # Remove the batch dimension if needed
        if x_output_tensor.shape[0] == 1:
            self.x_out_flat = Lambda(lambda t: K.squeeze(t, 0))(x_output_tensor)
        else:
            self.x_out_flat = x_output_tensor

        return x_input_tensors, x_output_tensor

    def in_out_tensors(self, multiplicity=None):
        """
        Builds the (inputs, outputs) tensors for the RGCN model. Currently supports only
        node-level tasks (i.e., classification or regression on a subset of nodes).

        Args:
            multiplicity (int, optional): How many target nodes each sample has.
                Defaults to the generator's multiplicity.

        Returns:
            (x_in, x_out):
            - x_in: a list of input tensors,
            - x_out: a tensor of shape (batch_size, layer_sizes[-1]) with final embeddings.
        """
        if multiplicity is None:
            multiplicity = self.multiplicity

        if multiplicity == 1:
            return self._node_model()
        else:
            raise NotImplementedError("Only node-level tasks are currently supported.")

    # Keep the same pattern of deprecating "build" in favor of "in_out_tensors"
    build = deprecated_model_function(in_out_tensors, "build")



class RelationalGraphConvolutionAttention(Layer):
    """
    A Keras layer implementing the Relational Graph Convolution Network (RGCNAttention) mechanism.

    Reference:
    Thomas N. Kipf & Michael Schlichtkrull (2017),
    "Modeling Relational Data with Graph Convolutional Networks"
    https://arxiv.org/pdf/1703.06103.pdf

    This layer expects:
      - A batch dimension of 1 (full-batch setup),
      - Node features as the first input,
      - Followed by one adjacency matrix per relationship.

    Args:
        units (int): Dimensionality of each node's transformed features.
        num_relationships (int): Number of distinct relationship types.
        num_bases (int): Number of basis matrices to factorize the relational parameters
            (use 0 or negative to disable basis decomposition).
        activation (str or callable): Activation function (e.g. 'relu', tf.nn.relu, ...).
        use_bias (bool): If True, includes a trainable bias term.
        final_layer (bool): Deprecated; use "tf.gather" or "GatherIndices" externally if needed.
        input_dim (int): (Optional) Size of the last axis of the node features if known in advance.
        kernel_initializer: Initializer for self and relational kernels (if no basis decomposition).
        kernel_regularizer: Regularizer for self and relational kernels.
        kernel_constraint: Constraint for self and relational kernels.
        bias_initializer: Initializer for the bias term.
        bias_regularizer: Regularizer for the bias term.
        bias_constraint: Constraint for the bias term.
        basis_initializer: Initializer for basis matrices if num_bases > 0.
        basis_regularizer: Regularizer for basis matrices.
        basis_constraint: Constraint for basis matrices.
        coefficient_initializer: Initializer for basis coefficients if num_bases > 0.
        coefficient_regularizer: Regularizer for basis coefficients.
        coefficient_constraint: Constraint for basis coefficients.
        kwargs: Standard Keras Layer keyword arguments.
    """

    def __init__(
            self,
            units,
            num_relationships,
            num_bases=0,
            activation=None,
            use_bias=True,
            final_layer=None,
            input_dim=None,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            kernel_constraint=None,
            bias_initializer="zeros",
            bias_regularizer=None,
            bias_constraint=None,
            basis_initializer="glorot_uniform",
            basis_regularizer=None,
            basis_constraint=None,
            coefficient_initializer="glorot_uniform",
            coefficient_regularizer=None,
            coefficient_constraint=None,
            **kwargs
    ):
        # If input_dim is known, pass it via input_shape to the parent class
        if "input_shape" not in kwargs and input_dim is not None:
            kwargs["input_shape"] = (input_dim,)

        # Initialize parent class
        super(RelationalGraphConvolutionAttention, self).__init__(**kwargs)

        # Validate arguments
        if not isinstance(units, int):
            raise TypeError("'units' must be an integer.")
        if units <= 0:
            raise ValueError("'units' must be a positive integer.")
        if not isinstance(num_relationships, int):
            raise TypeError("'num_relationships' must be an integer.")
        if num_relationships <= 0:
            raise ValueError("'num_relationships' must be > 0.")
        if not isinstance(num_bases, int):
            raise TypeError("'num_bases' must be an integer.")

        # Assign layer parameters
        self.units = units
        self.num_relationships = num_relationships
        self.num_bases = num_bases
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        # Manage user-provided or default config for kernel, bias, basis, coefficient
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.basis_initializer = initializers.get(basis_initializer)
        self.basis_regularizer = regularizers.get(basis_regularizer)
        self.basis_constraint = constraints.get(basis_constraint)

        self.coefficient_initializer = initializers.get(coefficient_initializer)
        self.coefficient_regularizer = regularizers.get(coefficient_regularizer)
        self.coefficient_constraint = constraints.get(coefficient_constraint)

        if final_layer is not None:
            raise ValueError(
                "The 'final_layer' argument is no longer supported. "
                "Use 'tf.gather' or 'GatherIndices' in your graph if needed."
            )

        # Call super constructor again (as done in the original)
        super(RelationalGraphConvolutionAttention, self).__init__(**kwargs)

    def get_config(self):
        """
        Returns a Python dictionary containing the layer configuration,
        enabling Keras to serialize/deserialize this layer.
        """
        cfg = super(RelationalGraphConvolutionAttention, self).get_config()
        cfg.update(
            {
                "units": self.units,
                "use_bias": self.use_bias,
                "activation": activations.serialize(self.activation),
                "kernel_initializer": initializers.serialize(self.kernel_initializer),
                "basis_initializer": initializers.serialize(self.basis_initializer),
                "coefficient_initializer": initializers.serialize(self.coefficient_initializer),
                "bias_initializer": initializers.serialize(self.bias_initializer),
                "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
                "basis_regularizer": regularizers.serialize(self.basis_regularizer),
                "coefficient_regularizer": regularizers.serialize(self.coefficient_regularizer),
                "bias_regularizer": regularizers.serialize(self.bias_regularizer),
                "kernel_constraint": constraints.serialize(self.kernel_constraint),
                "basis_constraint": constraints.serialize(self.basis_constraint),
                "coefficient_constraint": constraints.serialize(self.coefficient_constraint),
                "bias_constraint": constraints.serialize(self.bias_constraint),
                "num_relationships": self.num_relationships,
                "num_bases": self.num_bases,
            }
        )
        return cfg

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape from the provided input shapes.

        Args:
            input_shapes (list/tuple): The shapes of the inputs
                (the first for node features, then adjacency matrices).

        Returns:
            tuple: The shape of the output tensor.
        """
        features_shape, _ = input_shapes
        # We expect a batch dimension of size 1 for full-batch
        batch_dim = features_shape[0]
        out_dim = features_shape[1]
        return batch_dim, out_dim, self.units

    def build(self, input_shapes):
        """
        Constructs the layer's weights given the input shapes.

        Args:
            input_shapes (list/tuple): Shapes of the inputs
                (node features and adjacency matrices).
        """
        feat_shape = input_shapes[0]
        input_dim = int(feat_shape[-1])
        n_nodes = int(feat_shape[1]) # Because shape is (batch=1, N, F)

        # If basis decomposition is enabled
        if self.num_bases > 0:
            # Basis matrices: shape (input_dim, units, num_bases)
            self.bases = self.add_weight(
                shape=(input_dim, self.units, self.num_bases),
                initializer=self.basis_initializer,
                name="bases",
                regularizer=self.basis_regularizer,
                constraint=self.basis_constraint,
            )

            # Coefficients for each relationship (length = num_relationships)
            self.coefficients = []
            for r in range(self.num_relationships):
                coeffs = self.add_weight(
                    shape=(self.num_bases,),
                    initializer=self.coefficient_initializer,
                    name="coeff",
                    regularizer=self.coefficient_regularizer,
                    constraint=self.coefficient_constraint,
                )
                self.coefficients.append(coeffs)

            # We'll compute relational_kernels dynamically in call()
            self.relational_kernels = None

        else:
            # If no basis decomposition is used, we have distinct kernels for each relation
            self.bases = None
            self.coefficients = None
            self.relational_kernels = []
            for r in range(self.num_relationships):
                w_rel = self.add_weight(
                    shape=(input_dim, self.units),
                    initializer=self.kernel_initializer,
                    name="relational_kernels",
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                )
                self.relational_kernels.append(w_rel)

        # Self-connection kernel
        self.self_kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name="self_kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        # Optional bias
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None

        # >>>> NEW: trainable attention logits for each relation <<<<
        # Shape = (N, N) for each of the R relations
        # We'll later do a row-wise softmax over these to get valid attention coefficients.
        self.attention_weights = []
        for r in range(self.num_relationships):
            attn = self.add_weight(
                shape=(n_nodes, n_nodes),
                initializer="glorot_uniform",
                name=f"attention_logits_rel_{r}",
                regularizer=None,
                constraint=None,
            )
            self.attention_weights.append(attn)

        self.built = True

    def call(self, inputs):
        """
        Forward-pass: applies the RGCNAttention transformation.

        Args:
            inputs (list):
                - The first element is a node feature tensor of shape (1, N, F).
                - The following elements are adjacency matrices (one for each relationship),
                  each shape (N, N).

        Returns:
            tf.Tensor of shape (1, N, units) representing the updated node embeddings.
        """
        # Unpack inputs
        features, *As = inputs

        # Expecting the batch dimension to be 1
        batch_dim, n_nodes, _ = K.int_shape(features)
        if batch_dim != 1:
            raise ValueError("This layer requires a batch dimension of exactly 1.")

        # Remove the batch dimension to simplify operations
        features = K.squeeze(features, 0)

        # Contribution from self-kernel
        output = K.dot(features, self.self_kernel)

        # If bases are used, compute relational weights from them
        if self.relational_kernels is None:
            # Combine basis matrices with each relationship's coefficients
            # => 1 weight matrix per relationship
            relational_kernels = [
                tf.einsum("ijk,k->ij", self.bases, cff) for cff in self.coefficients
            ]
        else:
            relational_kernels = self.relational_kernels

        # >>>> Here is the main RGCNAttention loop with learned attention <<<<
        for i in range(self.num_relationships):
            tf.print("Relation", i, "shape(As[i]) =", tf.shape(As[i]))

            # 1) Row-wise softmax over the attention_logits
            #    This yields an (N, N) matrix alpha[i], where each row sums to 1.
            alpha = tf.nn.softmax(self.attention_weights[i], axis=1)  # shape (N, N)
            tf.print("Relation", i, "attention matrix alpha shape =", tf.shape(alpha))

            # 2) Elementwise multiply by adjacency As[i].
            #    The adjacency might be dense or dense-converted from sparse:
            A_tilde = alpha * As[i]  # shape (N, N)
            tf.print("Relation", i, "A_tilde shape =", tf.shape(A_tilde))

            # 3) Weighted neighbor aggregation
            h_graph = K.dot(A_tilde, features)  # shape (N, F)

            # 4) Apply the relation-specific kernel
            output += K.dot(h_graph, relational_kernels[i])

            tf.print("Done with RGCNAttention message passing, output shape =", tf.shape(output))

        # Bias if applicable
        if self.bias is not None:
            out_transformed = output + self.bias

        # Activation
        out_activated = self.activation(out_transformed)

        # Reintroduce the batch dimension
        out_final = K.expand_dims(out_activated, 0)

        return out_final


class RGCNAttention:
    """
    A relational GCN model, stacking multiple RelationalGraphConvolutionAttention layers
    to process graphs with multiple relationship (edge) types, as per
    "Modeling Relational Data with Graph Convolutional Networks"
    (Kipf & Schlichtkrull, 2017).

    Overview:
      - Takes a list of layer sizes (int) to define the hidden dimensions.
      - Has matching activation functions per layer.
      - Uses a `RelationalFullBatchNodeGenerator` to produce input tensors,
        including adjacency representations (sparse or dense).

    Notes:
      - The layer requires the batch dimension to be exactly 1 (full-batch).
      - Node ordering is consistent with the adjacency matrix; the final layer
        uses a gather operation to extract the requested node outputs.
      - For usage, see the `RelationalFullBatchNodeGenerator` documentation
        and the examples in the StellarGraph library.

    Example:
        .. code-block:: python

            gen = RelationalFullBatchNodeGenerator(G)
            rgcn_model = RGCNAttention(
                layer_sizes=[32, 4],
                activations=["elu","softmax"],
                num_bases=10,
                generator=gen,
                dropout=0.5
            )
            x_inp, out_prediction = rgcn_model.in_out_tensors()

    Args:
        layer_sizes (list of int): Dimensions of each RGCNAttention layer in the model.
        generator (RelationalFullBatchNodeGenerator): Data source; must be a
            `RelationalFullBatchNodeGenerator`.
        bias (bool): Whether to include a bias term in the RGCNAttention layers.
        num_bases (int): Number of basis matrices to use in decomposition; 0 or negative means none.
        dropout (float): Dropout rate applied before each RGCNAttention layer.
        activations (list of str or callable): Activation functions per layer. Defaults to 'relu' if None.
        kernel_initializer: Initializer for weights in each RGCNAttention layer.
        kernel_regularizer: Regularizer for weights in each RGCNAttention layer.
        kernel_constraint: Constraint for weights in each RGCNAttention layer.
        bias_initializer: Initializer for bias terms in each RGCNAttention layer.
        bias_regularizer: Regularizer for bias terms.
        bias_constraint: Constraint for bias terms.
    """

    def __init__(
            self,
            layer_sizes,
            generator,
            bias=True,
            num_bases=0,
            dropout=0.0,
            activations=None,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            kernel_constraint=None,
            bias_initializer="zeros",
            bias_regularizer=None,
            bias_constraint=None,
    ):
        # Validate the generator type
        if not isinstance(generator, RelationalFullBatchNodeGenerator):
            raise TypeError(
                "Expected generator to be a RelationalFullBatchNodeGenerator instance."
            )

        # Basic parameters
        self.layer_sizes = layer_sizes
        self.use_bias = bias
        self.num_bases = num_bases
        self.dropout_rate = dropout

        # Copy essential info from the generator
        self.multiplicity = generator.multiplicity
        self.n_nodes = generator.features.shape[0]
        self.num_features = generator.features.shape[1]
        self.num_relations = len(generator.As)
        self.use_sparse = generator.use_sparse

        # Number of layers to construct
        num_layers = len(layer_sizes)

        # Set or check activation functions
        if activations is None:
            activations = ["relu"] * num_layers
        elif len(activations) != num_layers:
            raise ValueError(
                "The number of activations must match the number of layers."
            )
        self.activations = activations
        self.num_bases = num_bases

        # Build up the stack of RGCNAttention layers + dropout
        self.model_layers = []
        for i in range(num_layers):
            self.model_layers.append(Dropout(self.dropout_rate))
            self.model_layers.append(
                RelationalGraphConvolutionAttention(
                    self.layer_sizes[i],
                    num_relationships=len(generator.As),
                    num_bases=self.num_bases,
                    activation=self.activations[i],
                    use_bias=self.use_bias,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_initializer=bias_initializer,
                    bias_regularizer=bias_regularizer,
                    bias_constraint=bias_constraint,
                )
            )

    def __call__(self, inputs):
        """
        Applies the RGCNAttention stack to the given inputs.

        Expected inputs for a full-batch setup:
            1) Node features, shape=(1, N, F)
            2) Output node indices, shape=(1, Z)
            3) Then adjacency data: either (indices, values) pairs if sparse,
               or a dense adjacency matrix for each relationship.

        Args:
            inputs (list): The input tensors for the model.

        Returns:
            A tensor containing the outputs for the requested nodes.
        """
        x_in, out_indices, *As = inputs

        # Validate that the batch dimension is 1
        b_dim, n_nodes, _ = K.int_shape(x_in)
        if b_dim != 1:
            raise ValueError("Full-batch RGCNAttention requires a batch dimension of exactly 1.")

        # For sparse adjacency
        if self.use_sparse:
            adj_indices_list = As[: self.num_relations]
            adj_values_list = As[self.num_relations:]
            # Convert each adjacency to a SqueezedSparseConversion
            final_As = [
                SqueezedSparseConversion(
                    shape=(n_nodes, n_nodes), dtype=adj_values_list[i].dtype
            )([adj_indices_list[i], adj_values_list[i]])
            for i in range(self.num_relations)
            ]
        else:
            # For dense adjacency, just remove the batch dimension
            final_As = [Lambda(lambda x: K.squeeze(x, 0))(A) for A in As]

        # Sequentially apply the layers
        h = x_in
        for layer in self.model_layers:
            # If it's an RGCNAttention layer, pass adjacency as well
            if isinstance(layer, RelationalGraphConvolutionAttention):
                h = layer([h] + final_As)
            else:
                h = layer(h)

        # Gather the results for the nodes of interest
        out_tensor = GatherIndices(batch_dims=1)([h, out_indices])
        return out_tensor

    def _node_model(self):
        """
        Assembles the RGCNAttention model for node-level tasks.

        Returns:
            (x_in, x_out):
            - x_in: list of input tensors (features, node indices, adjacency info),
            - x_out: the output tensor after applying the RGCNAttention stack.
        """

        # Inputs for the node features and the output indices
        x_in_input = Input(batch_shape=(1, self.n_nodes, self.num_features))
        out_indices_input = Input(batch_shape=(1, None), dtype="int32")

        if self.use_sparse:
            # Sparse adjacency placeholders
            adj_indices_list = [
                Input(batch_shape=(1, None, 2), dtype="int64")
                for _ in range(self.num_relations)
            ]
            adj_values_list = [
                Input(batch_shape=(1, None)) for _ in range(self.num_relations)
            ]
            adjacency_placeholders = adj_indices_list + adj_values_list
        else:
            # Dense adjacency placeholders
            adjacency_placeholders = [
                Input(batch_shape=(1, self.n_nodes, self.n_nodes))
                for _ in range(self.num_relations)
            ]

        # Combine all input tensors
        x_input_tensors = [x_in_input, out_indices_input] + adjacency_placeholders
        # Pass them through our model (via __call__)
        x_output_tensor = self(x_input_tensors)

        # Remove the batch dimension if needed
        if x_output_tensor.shape[0] == 1:
            self.x_out_flat = Lambda(lambda t: K.squeeze(t, 0))(x_output_tensor)
        else:
            self.x_out_flat = x_output_tensor

        return x_input_tensors, x_output_tensor

    def in_out_tensors(self, multiplicity=None):
        """
        Builds the (inputs, outputs) tensors for the RGCNAttention model. Currently supports only
        node-level tasks (i.e., classification or regression on a subset of nodes).

        Args:
            multiplicity (int, optional): How many target nodes each sample has.
                Defaults to the generator's multiplicity.

        Returns:
            (x_in, x_out):
            - x_in: a list of input tensors,
            - x_out: a tensor of shape (batch_size, layer_sizes[-1]) with final embeddings.
        """
        if multiplicity is None:
            multiplicity = self.multiplicity

        if multiplicity == 1:
            return self._node_model()
        else:
            raise NotImplementedError("Only node-level tasks are currently supported.")

    # Keep the same pattern of deprecating "build" in favor of "in_out_tensors"
    build = deprecated_model_function(in_out_tensors, "build")





