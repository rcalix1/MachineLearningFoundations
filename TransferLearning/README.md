## Phase 2: Transfer Learning

* IMPORTANT: https://github.com/rcalix1/TransferLearning/blob/main/algorithms/ResNetFromTorch_freezeLayers.ipynb
* VIT
* https://github.com/rcalix1/DeepLearningAlgorithms/blob/main/SecondEdition/Chapter11_TransferLearning/TransferLearn.pdf
* AlexNet: https://github.com/rcalix1/TransferLearning/tree/main/Vision
* Lena blurring to understand convolutions: https://github.com/rcalix1/TransferLearning/tree/main/algorithms/CNNs

## Phase 1:

* Data set is in this link:
* Also Examples of code for Phase 1
* https://github.com/rcalix1/MachineLearningFoundations/tree/main/NeuralNets/PyTorch


## Get ImageNet class string from pred label (e.g. 601)


```

import json
import urllib.request

# load mapping once
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

# example
idx = 601
print(classes[idx])
