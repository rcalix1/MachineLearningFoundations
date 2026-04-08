## Phase 1:

* Data set is in this link:
* Also Examples of code for Phase 1
* https://github.com/rcalix1/MachineLearningFoundations/tree/main/NeuralNets/PyTorch
* Lena blurring to understand convolutions: https://github.com/rcalix1/TransferLearning/tree/main/algorithms/CNNs


## Phase 2: Transfer Learning

* IMPORTANT: https://github.com/rcalix1/TransferLearning/blob/main/algorithms/ResNetFromTorch_freezeLayers.ipynb
* AlexNet: https://github.com/rcalix1/TransferLearning/tree/main/Vision
* https://github.com/rcalix1/DeepLearningAlgorithms/blob/main/SecondEdition/Chapter11_TransferLearning/TransferLearn.pdf

## Phase 2: Transfer Learning with Vision Transformer (ViT)

* VIT
* FruitsAdversarial.zip: https://github.com/rcalix1/CyberSecurityAndMachineLearning/tree/main/FirstEdition/Ch10_AIassurance/AdversarialML
* You can find the fruit zip files here: https://github.com/rcalix1/DeepLearningAlgorithms/tree/main/SecondEdition/chapter6_CNNs



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

```






