"""
Much of the following code has been copied over from: https://github.com/Newmu/dcgan_code
in order to make this file self-contained.

DCGAN code requires a CUDA GPU. In order to run the code on a machine without GPU:

1. Checkout DCGAN code from https://github.com/Newmu/dcgan_code

2. Make the following changes to mnist/train_cond_dcgan.py:
    a. Add the following imports:
        from theano.tensor.nnet import conv2d
        from theano.tensor.nnet import abstract_conv

    b. Add 'shit_no_gpu = True'

    c. Replace the functions gen(), discrim() with the following from this file:
        - conv_transpose()
        - gen()
        - discrim()

You should now be able to replicate the results of DCGAN without a gpu.
"""

import sys
sys.path.append('..')

import numpy as np
from numpy.random import RandomState
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import abstract_conv

from theano.sandbox.cuda.basic_ops import (gpu_contiguous, gpu_alloc_empty)

shit_no_gpu = True

seed = 42
np_rng = RandomState(seed)

k = 1             # # of discrim updates for each gen update
l2 = 2.5e-5       # l2 weight decay
b1 = 0.5          # momentum term of adam
nc = 1            # # of channels in image
ny = 10           # # of classes
nbatch = 128      # # of examples in batch
npx = 28          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngfc = 1024       # # of gen units for fully connected layers
ndfc = 1024       # # of discrim units for fully connected layers
ngf = 64          # # of gen filters in first conv layer
ndf = 64          # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 100       # # of iter at starting learning rate
niter_decay = 100 # # of iter to linearly decay learning rate to zero
lr = 0.0002       # initial learning rate for adam


class Rectify(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return (x + abs(x)) / 2.0

class LeakyRectify(object):

    def __init__(self, leak=0.2):
        self.leak = leak

    def __call__(self, x):
        f1 = 0.5 * (1 + self.leak)
        f2 = 0.5 * (1 - self.leak)
        return f1 * x + f2 * abs(x)

class Sigmoid(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return T.nnet.sigmoid(x)

class Normal(object):
    def __init__(self, loc=0., scale=0.05):
        self.scale = scale
        self.loc = loc

    def __call__(self, shape, name=None):
        return sharedX(np_rng.normal(loc=self.loc, scale=self.scale, size=shape), name=name)

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

def conv_cond_concat(x, y):
    """
    concatenate conditioning vector on feature map axis
    """
    return T.concatenate([x, y*T.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3]))], axis=1)

def batchnorm(X, g=None, b=None, u=None, s=None, a=1., e=1e-8):
    """
    batchnorm with support for not using scale and shift parameters
    as well as inference values (u and s) and partial batchnorm (via a)
    will detect and use convolutional or fully connected version
    """
    if X.ndim == 4:
        if u is not None and s is not None:
            b_u = u.dimshuffle('x', 0, 'x', 'x')
            b_s = s.dimshuffle('x', 0, 'x', 'x')
        else:
            b_u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            b_s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        if a != 1:
            b_u = (1. - a)*0. + a*b_u
            b_s = (1. - a)*1. + a*b_s
        X = (X - b_u) / T.sqrt(b_s + e)
        if g is not None and b is not None:
            X = X*g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')
    elif X.ndim == 2:
        if u is None and s is None:
            u = T.mean(X, axis=0)
            s = T.mean(T.sqr(X - u), axis=0)
        if a != 1:
            u = (1. - a)*0. + a*u
            s = (1. - a)*1. + a*s
        X = (X - u) / T.sqrt(s + e)
        if g is not None and b is not None:
            X = X*g + b
    else:
        raise NotImplementedError
    return X

def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    """
    sets up dummy convolutional forward pass and uses its grad as deconv
    currently only tested/working with same padding
    """
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img


relu = Rectify()
sigmoid = Sigmoid()
lrelu = LeakyRectify()
bce = T.nnet.binary_crossentropy

gifn = Normal(scale=0.02)
difn = Normal(scale=0.02)

gw  = gifn((nz+ny, ngfc), 'gw')
gw2 = gifn((ngfc+ny, ngf*2*7*7), 'gw2')
gw3 = gifn((ngf*2+ny, ngf, 5, 5), 'gw3')
gwx = gifn((ngf+ny, nc, 5, 5), 'gwx')

dw  = difn((ndf, nc+ny, 5, 5), 'dw')
dw2 = difn((ndf*2, ndf+ny, 5, 5), 'dw2')
dw3 = difn((ndf*2*7*7+ny, ndfc), 'dw3')
dwy = difn((ndfc+ny, 1), 'dwy')

gen_params = [gw, gw2, gw3, gwx]
discrim_params = [dw, dw2, dw3, dwy]


def conv_transpose(output, filters, output_shape, filter_size, subsample=(1, 1), border_mode=(0, 0)):
    """Compute convolution transpose (deconv)
    Note: We will assume zero padding, non-unit strides as explained in (Dumoulin, Visin):
        http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html

    Some notations:
        input_shape (batch size (b), input channels (c), input rows (i1), input columns (i2))
        filter_shape (output channels (c1), input channels (c2), filter rows (k1), filter columns (k2))

    Note: I am a bit confused by the theano code illustrations by (Dumoulin, Visin):

        input = theano.tensor.nnet.abstract_conv.conv2d_grad_wrt_inputs(
            ..., input_shape=(b, c1, o_prime1, o_prime2),
            filter_shape=(c1, c2, k1, k2), ...)

        The above seems misleading and input_shape=(b, c2, o_prime1, o_prime2)
        seems more appropriate (note the use of c2 instead of c1).
    """
    k1, k2 = (filter_size[0], filter_size[1])
    # we do not support even filter sizes
    assert (k1 % 2 == 1)
    assert (k2 % 2 == 1)
    # We are only considering the case where input_size is
    # such that a = (a1, a2) = (1, 1) = (1 + 2p - k) mod s
    # Note: by 'p' we mean the vector 'border_mode', by 's' we mean the vector 'subsample'.
    # a1, a2 in {0, ..., s-1}
    a1 = 1
    a2 = 1
    o_prime1 = subsample[0] * (output_shape[2] - 1) + a1 + k1 - 2 * border_mode[0]
    o_prime2 = subsample[1] * (output_shape[3] - 1) + a2 + k2 - 2 * border_mode[1]
    input_shape = (None, None, o_prime1, o_prime2)
    #print "a: (%d, %d)" % (a1, a2)
    #print "Filter size: (%d, %d)" % (k1, k2)
    #print "Input size: (%d, %d)" % (input_shape[2], input_shape[3])
    input = abstract_conv.conv2d_grad_wrt_inputs(
        output, filters, input_shape=input_shape, filter_shape=None,
        subsample=subsample, border_mode=border_mode)
    return input


def gen(Z, Y, w, w2, w3, wx):
    """Returns synthetic images for input random vectors.

    The objectives here are:
        1. Demystify the numbers behind the various filter sizes.
        2. Demonstrate the use of conv2d_grad_wrt_inputs for stacking
           multiple deconvolution (convolution transpose) layers
        3. Run DCGAN MNIST code without GPU.

    The functional intent is that when a random vector z of length nz (64) and
    its corresponding binary label vector y of length ny (10) is input,
    the output should be a 28x28 image with nc (1) channels.

    In reality, we feed in a batch of instances Z, Y of dimensions (nbatch, nz)
    and (nbatch, ny) respectively and get back a 4D tensor of shape (nbatch, nc, 28, 28)
    """
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    Z = T.concatenate([Z, Y], axis=1)
    h = relu(batchnorm(T.dot(Z, w)))
    h = T.concatenate([h, Y], axis=1)
    h2 = relu(batchnorm(T.dot(h, w2)))
    # at this point, h2.shape = (nbatch, ngf*2*7*7)
    h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
    h2 = conv_cond_concat(h2, yb)
    # at this point, h2.shape = (nbatch, ngf*2+ny, 7, 7)
    if shit_no_gpu:
        h3 = relu(batchnorm(conv_transpose(h2, w3, output_shape=(None, None, 7, 7),
                                           filter_size=(5, 5),
                                           subsample=(2, 2), border_mode=(2, 2))))
        # Since w3.shape = (ngf*2+ny, ngf, 5, 5), then if we follow the size
        # computations in conv_transpose(), we get:
        #   h3.shape = (h2.shape[0], ngf*2+ny, 2*(7-1)+5+1-2*2, 2*(7-1)+5+1-2*2)
        #            = (h2.shape[0], ngf*2+ny, 14, 14)
    else:
        h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
    h3 = conv_cond_concat(h3, yb)
    if shit_no_gpu:
        x = sigmoid(conv_transpose(h3, wx, output_shape=(None, None, 14, 14),
                                   filter_size=(5, 5),
                                   subsample=(2, 2), border_mode=(2, 2)))
        # Since wx.shape = (ngf+ny, nc, 5, 5), then if we follow the size
        # computations in conv_transpose(), we get:
        #   x.shape = (h3.shape[0], nc, 2*(14-1)+5+1-2*2, 2*(14-1)+5+1-2*2)
        #            = (h3.shape[0], nc, 28, 28)
    else:
        x = sigmoid(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x


def discrim(X, Y, w, w2, w3, wy):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    X = conv_cond_concat(X, yb)
    if shit_no_gpu:
        h = lrelu(conv2d(X, w, subsample=(2, 2), border_mode=(2, 2)))
    else:
        h = lrelu(dnn_conv(X, w, subsample=(2, 2), border_mode=(2, 2)))
    h = conv_cond_concat(h, yb)
    if shit_no_gpu:
        h2 = lrelu(batchnorm(conv2d(h, w2, subsample=(2, 2), border_mode=(2, 2))))
    else:
        h2 = lrelu(batchnorm(dnn_conv(h, w2, subsample=(2, 2), border_mode=(2, 2))))
    h2 = T.flatten(h2, 2)
    h2 = T.concatenate([h2, Y], axis=1)
    h3 = lrelu(batchnorm(T.dot(h2, w3)))
    h3 = T.concatenate([h3, Y], axis=1)
    y = sigmoid(T.dot(h3, wy))
    return y

X = T.tensor4()
Z = T.matrix()
Y = T.matrix()

gX = gen(Z, Y, *gen_params)

p_img = discrim(X, Y, *discrim_params)
p_gen = discrim(gX, Y, *discrim_params)

zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))
ymb = floatX(np_rng.randint(0, 1, size=(nbatch, ny)))

gen_img = theano.function([Z, Y], gX)
p1 = theano.function([X, Y], p_img)
p2 = theano.function([Z, Y], p_gen)

tmp = gen_img(zmb, ymb)
print "generated synthetic with shape %s" % str(tmp.shape)

# since tmp is a batch of 28x28 images with 1 channel,
# discrim() should be able to process them without errors.
# Validate that this happens.
tmp_p = p1(tmp, ymb)
print "discriminator output shape %s" % str(tmp_p.shape)

# Validate that we can tie the generator output with discriminator
# input to ensure DCGAN works correctly.
tmp_p = p2(zmb, ymb)
print "discriminator output shape %s" % str(tmp_p.shape)

