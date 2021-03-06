# https://github.com/facebookresearch/ResNeXt/blob/master/README.md


import os
from torch.autograd import Variable

#--------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from functools import reduce

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))





class ResNext(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(ResNext, self).__init__()
        in_channels, height, width = in_shape

        self.resnext_101_64x4d = nn.Sequential( # Sequential,
            nn.Conv2d(in_channels,64,(7, 7),(2, 2),(3, 3),1,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3),(2, 2),(1, 1)),
            nn.Sequential( # Sequential,
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(256),
                        ),
                        nn.Sequential( # Sequential,
                            nn.Conv2d(64,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(256),
                        ),
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(256),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(256,256,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(256),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
            ),
            nn.Sequential( # Sequential,
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(256,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512,512,(3, 3),(2, 2),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(512),
                        ),
                        nn.Sequential( # Sequential,
                            nn.Conv2d(256,512,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(512),
                        ),
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(512),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(512),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(512,512,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(512),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
            ),
            nn.Sequential( # Sequential,
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(512,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(2, 2),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        nn.Sequential( # Sequential,
                            nn.Conv2d(512,1024,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                                nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(1024),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(1024,1024,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(1024),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
            ),
            nn.Sequential( # Sequential,
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(1024,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(2048),
                                nn.ReLU(),
                                nn.Conv2d(2048,2048,(3, 3),(2, 2),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(2048),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(2048,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(2048),
                        ),
                        nn.Sequential( # Sequential,
                            nn.Conv2d(1024,2048,(1, 1),(2, 2),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(2048),
                        ),
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(2048,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(2048),
                                nn.ReLU(),
                                nn.Conv2d(2048,2048,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(2048),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(2048,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(2048),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
                nn.Sequential( # Sequential,
                    LambdaMap(lambda x: x, # ConcatTable,
                        nn.Sequential( # Sequential,
                            nn.Sequential( # Sequential,
                                nn.Conv2d(2048,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                                nn.BatchNorm2d(2048),
                                nn.ReLU(),
                                nn.Conv2d(2048,2048,(3, 3),(1, 1),(1, 1),1,64,bias=False),
                                nn.BatchNorm2d(2048),
                                nn.ReLU(),
                            ),
                            nn.Conv2d(2048,2048,(1, 1),(1, 1),(0, 0),1,1,bias=False),
                            nn.BatchNorm2d(2048),
                        ),
                        Lambda(lambda x: x), # Identity,
                    ),
                    LambdaReduce(lambda x,y: x+y), # CAddTable,
                    nn.ReLU(),
                ),
            ),
            nn.AdaptiveAvgPool2d(1),
            Lambda(lambda x: x.view(x.size(0),-1)), # View,
            nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2048,num_classes)), # Linear,
        )

    def forward(self, x):
        x = self.resnext_101_64x4d(x)
        logit = x
        prob  = F.sigmoid(logit)
        return logit,prob

########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 16
    num_classes = 17
    C,H,W = 3,224,224

    inputs = torch.randn(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)
    in_shape = inputs.size()[1:]

    if 1:
        net = ResNext(in_shape=in_shape, num_classes=num_classes).cuda().train()
        #densenet121


        x = Variable(inputs)
        logits, probs = net.forward(x.cuda())

        loss = nn.MultiLabelSoftMarginLoss()(logits, Variable(labels.cuda()))
        loss.backward()

        print(type(net))
        print(net)

        print('probs')
        print(probs)

        #input('Press ENTER to continue.')

