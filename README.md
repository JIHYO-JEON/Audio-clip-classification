# Audio Clip Classification with Convolutional Neural Network

Audio clip classification with AlexNet, GoogLeNet and VGGNet


A lot of convolutional neural network models were presented in the annual ImageNet Large Scale Visual Recognition Challenge (ILSVRC). The computer vision is the easiest way to evaluate the models. In this paper, the neural network is built to classify audio clips. There are 10 audio clip classes. The VGGNet-like model and the AlexNet-like model were used as the training model. In this experiment setup, the AlexNet-like model shows the better result than the VGGNet-like model. In the case of AlexNet-like model, the loss value after 100 iterations was 0.004.

The VGGNet-like architecture and the AlexNet-like architecture is used for classifying audio clips. Those are trained with the audio clip dataset Urbansound8K. The AlexNet-like model shows the better result than the VGGNet-like model. The cost value of the AlexNet-like model after 100 iterations was 0.004. There are some possible reasons such as too many parameters and vanishing gradient problem.
	Any unsupervised pre-training was not done in this model so it can be better if the model is pre-trained. Also, making the network larger, deeper, and further training would likely improve the result.
