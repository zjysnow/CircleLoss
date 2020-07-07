
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock
from mxnet.gluon.data import DataLoader
import mxboard
import time

import glob
import os

from .circle_loss import CircleLoss
from .similarity import Similarity

ctx = mx.gpu(3)
#batch_size = 80
batch_size = 52
nepoch = 100
nclass = 8631 # VGGFace2 Database

def get_imglist(list_file):
    data = list(open(list_file, 'r'))
    imglist = []
    for d in data:
        name, label = d.split()
        imglist.append([float(label), name])

    return imglist

train_loader = mx.image.ImageIter(batch_size=batch_size, data_shape=(3,112,112), 
                                  imglist=get_imglist('./list.8631.txt'), 
                                  path_root='VGGFace2', shuffle=True, last_batch_handle='roll_over')

class NetH(HybridBlock):
    def __init__(self, json, nclass, ctx=mx.cpu()):
        super(NetH, self).__init__()
        with self.name_scope():
            self.insight = nn.SymbolBlock.imports(symbol_file=json, input_names=['data'], ctx=ctx)
            self.dense = Similarity(units=nclass, in_units=512, use_bias=False)

    def hybrid_forward(self, F, x):
        x = self.insight(x)
        x = self.dense(x)
        return x

def train1():
    neth = NetH(json='insightface.json', nclass=nclass, ctx=ctx)
    neth.hybridize()
    trainer_neth = gluon.Trainer(neth.collect_params(), 'adam', {'learning_rate': 1e-4})

    circle_loss = CircleLoss(nclass, scale=64, margin=0.25)

    tick = time.time()
    ts = time.localtime(tick)
    stamp = time.strftime('%Y%m%d%H%M%S', ts)
    with mxboard.SummaryWriter(logdir='logs/'+ stamp) as sw:
        iternum = 0
        for epoch in range(nepoch):
            for batch in train_loader:
                faces = batch.data[0].as_in_context(ctx) # 0-1
                labels = batch.label[0].as_in_context(ctx)

                with autograd.record():
                    pred = neth(faces)
                    loss = circle_loss(pred, labels)
                    loss.backward()
                trainer_neth.step(batch_size)

                if iternum % 100 == 0:
                    step = iternum / 100
                    print("epoch: %d, iter: %d, loss: %f"%(epoch, iternum, loss.mean().asscalar()))
                    sw.add_scalar("LOSS", value=('LOSS',loss.mean().asscalar()), global_step=step)
                    pass
                
                iternum = iternum + 1
                
                pass # for in train_loader
            neth.export('circle', epoch)
            

if __name__ == '__main__':
    train1()
    pass