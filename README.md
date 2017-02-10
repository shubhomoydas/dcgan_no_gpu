# dcgan_no_gpu
Replace parts of DCGAN code (importantly, deconv with conv2d_grad_wrt_inputs) so that it runs without GPU
------------

So ... you tried to run DCGAN MNIST (https://github.com/Newmu/dcgan_code) and soon realized: Shit! I don't have a GPU!! dnn_conv() is easy to replace; but the real offending bit is deconv(). As workaround you tried replacing the gen() architecture with simpler mlp. But ... the results are shitty! You *really* want the deconv() or its equivalent. Next, you googled and found there *is* a way (actually, one of many): conv2d_grad_wrt_inputs(). However, documentation about that is quite confusing. A helpful guide (http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html) does exist, but getting to the 'Transposed Convolution' bit leaves you more confused. What is the 'input' and what is the 'output' when looking bottom up from the generator? How do we stack two conv transpose layers? How do the filter sizes magically result in generating 28x28 images?

Well, let's see if the code here helps.


To execute the code:
------------
pythonw -m mnist.mnist_train_cond_dcgan_no_gpu


Modifying the original DCGAN Code
------------
Much of the code in mnist_train_cond_dcgan_no_gpu.py has been copied over from: https://github.com/Newmu/dcgan_code
in order to make it self-contained. This file (mnist_train_cond_dcgan_no_gpu.py) only illustrates the small bits you need to replace for DCGAN to work without GPU. To run full DCGAN, you *will* need to get the entire original code first. Follow these steps:

1. Checkout DCGAN code from https://github.com/Newmu/dcgan_code

2. Make the following changes to mnist/train_cond_dcgan.py:

    a. Add the following imports:
    
        from theano.tensor.nnet import conv2d
        from theano.tensor.nnet import abstract_conv

    b. Add 'shit_no_gpu = True'

    c. Replace the functions gen(), discrim() with the following from mnist_train_cond_dcgan_no_gpu.py:
    
        - conv_transpose()
        - gen()
        - discrim()

You should now be able to replicate the results of DCGAN without a gpu when you run the modified code.

If you are already familiar with DCGAN, then cut through all other stuff in the code and focus on only three:

    conv_transpose(): This is where lies the real meat of this code. Demonstrates how to apply convolution transform (deconv) using conv2d_grad_wrt_inputs.
    
    gen(): Shows where/how to replace the GPU dependency. The other important information is the input/output size calculation.
    
    discrim(): Contains simple replacement of dnn_conv with conv2d.

Note: You will see that we do not load any actual images in this code. That is not the point here. We only provide non-gpu implementations of gpu-dependent functions and validate that they conform with the expected input/output sizes.
