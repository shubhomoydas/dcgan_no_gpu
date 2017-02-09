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

You should now be able to replicate the results of DCGAN without a gpu.
