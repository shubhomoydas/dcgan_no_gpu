# dcgan_no_gpu
Replace parts of DCGAN code (importantly, deconv with conv2d_grad_wrt_inputs) so that it runs without GPU
------------

So ... you tried to run DCGAN MNIST (https://github.com/Newmu/dcgan_code) and soon realized: Shit! I don't have a GPU!! dnn_conv() is easy to replace; but the real offending bit is deconv(). As workaround you could try replacing the gen() architecture with a simpler MLP. But the results would likely remain bad for many epochs. You *really* want the deconv() or its equivalent.

Fortunately, Theano *does* support convolutional transpose (a.k.a deconv) with the more accurately named API conv2d_grad_wrt_inputs(). However, documentation on it is a bit confusing despite the presence of a [very] helpful guide (http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html). You need to wrap your head around: (a) what is the 'input' and what is the 'output' when looking bottom up from the generator's [random] input layer? (b) how do we stack two convolutional transpose layers? (c) how do the filter sizes magically result in generating 28x28 images? I hope the code here answers these questions.


To execute the code:
------------
pythonw -m mnist.mnist_train_cond_dcgan_no_gpu


Modifying the original DCGAN Code
------------
Much of the code in mnist_train_cond_dcgan_no_gpu.py has been copied over from: https://github.com/Newmu/dcgan_code
in order to make it self-contained. This file only illustrates the small bits you need to replace for DCGAN to work without GPU. To run full DCGAN, you *will* need to checkout the entire original code first. Follow these steps:

1. Checkout DCGAN code from https://github.com/Newmu/dcgan_code

2. Make the following changes to mnist/train_cond_dcgan.py:

    a. Add the imports:
    
        from theano.tensor.nnet import conv2d
        from theano.tensor.nnet import abstract_conv

    b. Declare 'shit_no_gpu = True'

    c. Replace the functions gen(), discrim() with the following from mnist_train_cond_dcgan_no_gpu.py:
    
    conv_transpose(): This is where lies the real meat of this code.
        Demonstrates how to apply convolution transpose (deconv) 
        using conv2d_grad_wrt_inputs.
    
    gen(): Shows where/how to replace the GPU dependency.
        The other important information is the input/output size calculation.
    
    discrim(): Contains simple replacement of dnn_conv with conv2d.

All other stuff in mnist_train_cond_dcgan_no_gpu.py (other than the bits mentioned above) can be ignored. With these modifications, you should be able to replicate the results of DCGAN without a gpu.

Note: The code here only provides non-gpu support for gpu-dependent functions and validates that the layers conform to the expected input/output sizes.
