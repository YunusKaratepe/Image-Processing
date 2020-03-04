import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob

IMG_SIZE = 128
REBUILD_DATA = False  # don't set this true if you don't want to rebuild your database


class Data:
    ALL_SOULS = "data/all_souls"
    ASHMOLEAN = "data/ashmolean"
    BALLIOL = "data//balliol"
    BODLEIAN = "data//bodleian"
    CHRIST_CHURCH = "data/christ_church"
    CORNMARKET = "data/cornmarket"
    HERTFORD = "data/hertford"
    JESUS = "data/jesus"
    KEBLE = "data/keble"
    MAGDALEN = "data/magdalen"
    NEW = "data/new"
    ORIEL = "data/oriel"
    OXFORD = "data/oxford"
    PITT_RIVERS = "data/pitt_rivers"
    RADCLIFFE_CAMERA = "data/radcliffe_camera"
    TRINITY = "data/trinity"
    WORCESTER = "data/worcester"
    LABELS = {ALL_SOULS: 0, ASHMOLEAN: 1, BALLIOL: 2, BODLEIAN: 3, CHRIST_CHURCH: 4, CORNMARKET: 5, HERTFORD: 6,
              JESUS: 7, KEBLE: 8, MAGDALEN: 9, NEW: 10, ORIEL: 11, OXFORD: 12, PITT_RIVERS: 13, RADCLIFFE_CAMERA: 14,
              TRINITY: 15, WORCESTER: 16}
    data = []

    def make_data(self):

        for label in self.LABELS:
            i = 0
            for path in tqdm(glob.glob(label + '/*.jpg')):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                self.data.append([np.array(img), np.eye(17)[self.LABELS[label]]])

        np.random.shuffle(self.data)
        np.save("data.npy", self.data)
        print(len(self.data))


if REBUILD_DATA:
    data = Data()
    data.make_data()

data = np.load("data.npy", allow_pickle=True)
# PART 2: CREATING A MODEL


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        x = torch.randn(IMG_SIZE, IMG_SIZE).view(-1, 1, IMG_SIZE, IMG_SIZE)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        return F.softmax(x, dim=1)


net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in data]).view(-1, IMG_SIZE, IMG_SIZE)
X = X / 255.0
y = torch.Tensor([i[1] for i in data])

VAL_PCT = 0.2
val_size = int(len(X) * VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]
test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X))
print(len(test_X))


BATCH_SIZE = 64
EPOCHS = 1


for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_X = train_X[i: i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
        batch_y = train_y[i: i + BATCH_SIZE]

        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(loss)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, IMG_SIZE, IMG_SIZE))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1

training_fc = []
test_fc = []

net = torch.load('model.pt')
with torch.no_grad():
    for i in tqdm(range(len(train_X))):
        net_out = net(train_X[i].view(-1, 1, IMG_SIZE, IMG_SIZE))[0]
        a = net_out.numpy()
        training_fc.append(a)

training_fc = np.asarray(training_fc)
print(training_fc.shape)

with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        net_out = net(test_X[i].view(-1, 1, IMG_SIZE, IMG_SIZE))[0]
        a = net_out.numpy()
        test_fc.append(a)

test_fc = np.asarray(test_fc)
print(test_fc.shape)

'''for i in tqdm(range(0, 1012)):
    min = 99999.0
    for j in range(0, 4051):
        value = 0.0
        for k in range(0, 512):
            value = value + (abs(test_fc[i][k] - training_fc[j][k]))
        if (value < min):
            min = value
            min_index = j
    f = open("result.txt", "a")
    f.write("Test datasindaki " + str(i) + ". resim ---> Train datasindaki " + str(min_index) + ". resme en yakindir.\n")
    f.close()'''





