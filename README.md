# Knowledge Distillation Net with Swish
Main files you may be interested in are the following:
- student.py: student network, training and testing with ReLu
- student_swish.py: student network, training and testing with Swish
- resnet.py: teacher network, training and testing with Swish
- swish_function.cpp: C++ implementation of the Swish Activation Function
- autograd_check.py: check the Swish implementation with torch.autograd.gradcheck
- kd.py: student network, training and testing with Swish and KD (both on the fly and disk caching)
- plots/: all the plots of the loss, top1 and top5 accuracy generated by plot.py
- log/: all the training and evaluation logs

## Network Structures
The student network has two conv layers with output channel number of 64 and 128 respectively. Each conv layer is also bundled with a max pooling layer with kernel size of 2 and stride of 2, a BN layer and an activation layer. After the conv layers two fc layers are applied.
	The teacher network has the same structure as ResNet18, with activation layers replaced with Swish activation and each output channel number reduced by half. I have also experimented with the original output channel numbers, but those experiments are very time consuming on CPUs.
	Cross-entropy is used in all experiments.
Swish Activation Implementation
	I implemented the Swish activation function in C++ based on the paper provided in the instruction. Since beta should be set to be trainable, I also calculated the derivative of beta, which is x^2*sigmoid(beta*x)*(1-sigmoid(beta*x)).
Comparison of Student Network with ReLu and Swish
Training and Hyperparameter Tuning

## Reference
https://arxiv.org/pdf/1710.05941.pdf
https://arxiv.org/abs/1503.02531
https://pytorch.org/tutorials/advanced/cpp_extension.html
https://github.com/pytorch/examples/tree/master/mnist 
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
https://github.com/peterliht/knowledge-distillation-pytorch 
