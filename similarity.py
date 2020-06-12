
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn
from mxnet import ndarray as nd

class Similarity(HybridBlock):
    def __init__(self, units, activation=None, use_bias=True, flatten=True,
                 dtype='float32', weight_initializer=None, bias_initializer='zeros',
                 in_units=0, **kwargs):
        super(Similarity, self).__init__(**kwargs)
        self._flatten = flatten
        with self.name_scope():
            self._units = units
            self._in_units = in_units
            self.weight = self.params.get('weight', shape=(units, in_units),
                                          init=weight_initializer, dtype=dtype,
                                          allow_deferred_init=True)
            if use_bias:
                self.bias = self.params.get('bias', shape=(units,),
                                            init=bias_initializer, dtype=dtype,
                                            allow_deferred_init=True)
            else:
                self.bias = None
            if activation is not None:
                self.act = nn.Activation(activation, prefix=activation+'_')
            else:
                self.act = None

    def hybrid_forward(self, F, x, weight, bias=None):
        x = F.L2Normalization(x)
        weight = F.L2Normalization(weight)
        act = F.FullyConnected(x, weight, bias, no_bias=bias is None, num_hidden=self._units,
                               flatten=self._flatten, name='fwd')
        if self.act is not None:
            act = self.act(act)
        return act