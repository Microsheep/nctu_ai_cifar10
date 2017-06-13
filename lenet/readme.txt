Implements Cifar10 using lenet in pytorch!

===========================

Decide if use trained net!
load = True/False

===========================

CPU will take more time if you want to try.
GPU takes about 3 minutes to go through 50 epoch.
(Training on Single GTX 1080)

===========================

Accuracy of the network on the 10000 test images: 76 %
Accuracy of plane : 75 %
Accuracy of   car : 89 %
Accuracy of  bird : 69 %
Accuracy of   cat : 58 %
Accuracy of  deer : 66 %
Accuracy of   dog : 60 %
Accuracy of  frog : 83 %
Accuracy of horse : 68 %
Accuracy of  ship : 84 %
Accuracy of truck : 84 %

===========================

Also add random horizontal flip and random crop (move the input)
Changed the mean and dev to what it shall be (about 0.5 and 0.2)
Auto decrease learning rate

