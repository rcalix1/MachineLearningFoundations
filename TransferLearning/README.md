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

```



## UNET for masking


```


# =========================
# SIMPLE U-NET SEGMENTATION (MNIST)
# =========================

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# DATA
# =========================
transform = T.ToTensor()

train_ds = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

# =========================
# CREATE SEGMENTATION TARGET
# =========================
# mask = 1 where pixel > threshold

def make_mask(x):
    return (x > 0.1).float()

# =========================
# SIMPLE U-NET
# =========================
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # down
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )

        # bottleneck
        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

        # up
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU()
        )

        # output mask
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)

        m = self.mid(e2)

        u = self.up(m)
        cat = torch.cat([u, e1], dim=1)

        d = self.dec1(cat)

        return torch.sigmoid(self.out(d))  # mask [0,1]

# =========================
# MODEL
# =========================
model = SimpleUNet().to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# =========================
# TRAIN
# =========================
EPOCHS = 3

for epoch in range(EPOCHS):
    for xb, _ in train_dl:
        xb = xb.to(device)

        mask = make_mask(xb)

        pred = model(xb)

        loss = loss_fn(pred, mask)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch} loss: {loss.item():.4f}")

# =========================
# TEST + VISUALIZE
# =========================
model.eval()

xb, _ = next(iter(train_dl))
xb = xb.to(device)

with torch.no_grad():
    pred = model(xb)

# =========================
# PLOT RESULTS
# =========================
plt.figure(figsize=(9,3))

for i in range(3):
    # original
    plt.subplot(3,3, i+1)
    plt.imshow(xb[i,0].cpu(), cmap='gray')
    plt.title("Input")
    plt.axis('off')

    # ground truth
    plt.subplot(3,3, i+4)
    plt.imshow(make_mask(xb[i]).cpu()[0], cmap='gray')
    plt.title("Mask")
    plt.axis('off')

    # prediction
    plt.subplot(3,3, i+7)
    plt.imshow(pred[i,0].cpu(), cmap='gray')
    plt.title("Pred")
    plt.axis('off')

plt.tight_layout()
plt.show()



```

