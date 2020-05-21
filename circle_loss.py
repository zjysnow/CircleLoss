
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import Block
#from mxnet.gluon import HybridBlock
#from mxnet.gluon.loss import Loss
from mxnet import ndarray as nd

class CircleLoss(Block):
    def __init__(self, nclass, scale=64, margin=0.25):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.nclass= nclass
        self.delta_p = 1 - margin
        self.delta_n = margin
        self.op = 1 + margin
        self.on = -margin
        self.loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
        pass

    def forward(self, pred, label):
        label = nd.one_hot(label, self.nclass)
        alpha_p = nd.relu(self.op - pred)
        alpha_n = nd.relu(pred - self.on)

        pred = (label * (alpha_p * (pred - self.delta_p)) + (1-label) * (alpha_n * (pred - self.delta_n))) * self.scale

        return self.loss(pred, label)
    pass

if __name__ == '__main__':
    batch_size = 10
    circle_loss = CircleLoss(nclass=10)
    labels = nd.array([1, 2, 3])
    feats = nd.array([[8.5675e-01, 1.5308e-01, 4.8325e-01],
                      [5.2465e-01, 7.8845e-01, 6.2924e-02],
                      [5.0915e-01, 7.7957e-01, 5.3919e-01],
                      [5.3219e-02, 7.3506e-01, 8.0188e-01],
                      [7.9237e-01, 8.4039e-01, 6.0384e-02],
                      [4.5711e-01, 5.0027e-01, 2.2803e-01],
                      [1.1671e-01, 9.6149e-01, 1.1187e-01],
                      [2.6528e-01, 7.8812e-01, 6.6587e-01],
                      [7.5084e-04, 5.8510e-01, 6.3166e-01],
                      [2.9980e-01, 2.8037e-01, 3.5707e-01]]).reshape(3,10)

    # 4x512
    loss = circle_loss(feats, labels)
    print(loss)