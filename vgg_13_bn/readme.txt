Implements Cifar10 using VGG_13_bn in pytorch!

===========================

Decide if use trained net!
load = True/False

===========================

CPU will take forever if you want to try...
GPU takes about 3 hours to go through 300 epoch...
(Training on Single GTX 1080)

===========================

Accuracy of the network on the 10000 test images: 87 %
Accuracy of plane : 86 %
Accuracy of   car : 92 %
Accuracy of  bird : 78 %
Accuracy of   cat : 64 %
Accuracy of  deer : 92 %
Accuracy of   dog : 90 %
Accuracy of  frog : 91 %
Accuracy of horse : 84 %
Accuracy of  ship : 96 %
Accuracy of truck : 87 %

===========================

Add random horizontal flip and random crop (move the input)
Changed the mean and dev to what it shall be (about 0.5 and 0.2)
Auto decrease learning rate

