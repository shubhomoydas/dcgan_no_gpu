# dcgan_no_gpu
Replace parts of DCGAN code (importantly, deconv) so that it runs without GPU
----------

So ... you tried to run DCGAN MNIST (https://github.com/Newmu/dcgan_code) and soon realized: Shit! I don't have a GPU!! dnn_conv() is easy to replace; but the real offending bit is deconv(). As workaround you tried replacing the gen() architecture with simpler mlp. But ... the results are shitty! You *really* want the deconv() or its equivalent. Next, you googled and found there *is* a way (actually, one of many): conv2d_grad_wrt_inputs(). However, documentation about that is quite confusing. A helpful guide (http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html) does exist, but getting to the 'Transposed Convolution' bit leaves you more confused. What is the 'input' and what is the 'output' when looking bottom up from the generator? How do we stack two conv transpose layers? How do the filter sizes magically result in generating 28x28 images?

Well, let's see if the code here helps.
