Implements Cifar10 using lenet like structure in pytorch!

===========================

Decide if use trained net!
load = True/False

===========================

CPU will take more time if you want to try.
GPU takes about 13 minutes to go through 200 epoch.
(Training on Single GTX 1080)

===========================

Accuracy of the network on the 10000 test images: 82 %
Accuracy of plane : 82 %
Accuracy of   car : 92 %
Accuracy of  bird : 75 %
Accuracy of   cat : 55 %
Accuracy of  deer : 59 %
Accuracy of   dog : 78 %
Accuracy of  frog : 86 %
Accuracy of horse : 80 %
Accuracy of  ship : 93 %
Accuracy of truck : 89 %

===========================

Think of it as just a increased channel lenet
Also add random horizontal flip and random crop (move the input)
Changed the mean and dev to what it shall be (about 0.5 and 0.2)
Auto decrease learning rate

