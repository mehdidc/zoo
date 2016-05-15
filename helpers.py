from lasagne import layers


class GramMatrixLayer(layers.Layer):
    def __init__(self, incoming,
                 **kwargs):
        super(GramMatrixLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        assert input.ndim == 4
        x = input
        g = (x[:, :, None, :, :] * x[:, None, :, :, :]).sum(axis=(3, 4))
        g = g.reshape((g.shape[0], -1))
        return g

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1] * input_shape[1])
