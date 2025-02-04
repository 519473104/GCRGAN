import torch.nn as nn
import torch
import random
from audtorch.metrics.functional import concordance_cc

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label = nn.Sequential(nn.Linear(1, 128),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.InstanceNorm1d(128),
                                   nn.ReLU())

        self.main = nn.Sequential(nn.Conv1d(1, 1, 3, 4, 1),
                                  nn.BatchNorm1d(1),
                                  nn.ReLU(),
                                  nn.Conv1d(1, 1, 3, 4, 1),
                                  nn.InstanceNorm1d(1),
                                  nn.ReLU())

        self.output = nn.Sequential(nn.Flatten(),
                                    nn.Linear(72, 10),
                                    nn.BatchNorm1d(10),
                                    nn.ReLU(),
                                    nn.Linear(10, 1),
                                    nn.BatchNorm1d(1),
                                    nn.Sigmoid())

    def forward(self, label, realdata):
        label = self.label(label)
        inputs = torch.cat((realdata, label), dim=1).unsqueeze(1)
        features = self.main(inputs)
        outputs = self.output(features)
        return outputs


# 生成器
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label = nn.Sequential(nn.Linear(1, 128),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.InstanceNorm1d(128),
                                   nn.ReLU())

        self.noise = nn.Sequential(nn.Linear(1024, 128),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.InstanceNorm1d(128),
                                   nn.ReLU())

        self.main = nn.Sequential(nn.Conv1d(1, 1, 3, 1, 1),
                                  nn.BatchNorm1d(1),
                                  nn.ReLU(),
                                  nn.Flatten(),
                                  nn.Linear(256, 512),
                                  nn.InstanceNorm1d(512),
                                  nn.ReLU(),
                                  nn.Linear(512, 1024))


    def candidate(self, data, y, num):

        new_data_0 = torch.zeros((num // 5, 1024))
        new_data_1 = torch.zeros((num // 5, 1024))
        new_data_2 = torch.zeros((num // 5, 1024))
        new_data_3 = torch.zeros((num // 5, 1024))
        new_data_4 = torch.zeros((num // 5, 1024))

        for j in range(5):
            train_data_0 = data[0]
            train_data_1 = data[num // 5]
            train_data_2 = data[num * 2 // 5]
            train_data_3 = data[num * 3 // 5]
            train_data_4 = data[num * 4 // 5]

            for i in range(len(data) // 5 - 1):
                index = random.randint(0, len(data) // 5 - 1)
                train_data_0 = torch.cat((train_data_0, data[index]))
            for i in range(len(data) // 5, len(data) * 2 // 5 - 1):
                index = random.randint(len(data) // 5, len(data) * 2 // 5 - 1)
                train_data_1 = torch.cat((train_data_1, data[index]))
            for i in range(len(data) * 2 // 5, len(data) * 3 // 5 - 1):
                index = random.randint(len(data) * 2 // 5, len(data) * 3 // 5 - 1)
                train_data_2 = torch.cat((train_data_2, data[index]))
            for i in range(len(data) * 3 // 5, len(data) * 4 // 5 - 1):
                index = random.randint(len(data) * 3 // 5, len(data) * 4 // 5 - 1)
                train_data_3 = torch.cat((train_data_3, data[index]))
            for i in range(len(data) * 4 // 5, len(data) * 5 // 5 - 1):
                index = random.randint(len(data) * 4 // 5, len(data) * 5 // 5 - 1)
                train_data_4 = torch.cat((train_data_4, data[index]))

            for i in range((num // 25) * j, (num // 25) * (j + 1)):
                index = random.randint(0, len(train_data_0) - 2000)
                new_data_0[i] = train_data_0[index:index + 1024]
            for i in range((num // 25) * j, (num // 25) * (j + 1)):
                index = random.randint(0, len(train_data_1) - 2000)
                new_data_1[i] = train_data_1[index:index + 1024]
            for i in range((num // 25) * j, (num // 25) * (j + 1)):
                index = random.randint(0, len(train_data_2) - 2000)
                new_data_2[i] = train_data_2[index:index + 1024]
            for i in range((num // 25) * j, (num // 25) * (j + 1)):
                index = random.randint(0, len(train_data_3) - 2000)
                new_data_3[i] = train_data_3[index:index + 1024]
            for i in range((num // 25) * j, (num // 25) * (j + 1)):
                index = random.randint(0, len(train_data_4) - 2000)
                new_data_4[i] = train_data_4[index:index + 1024]

        new_data = torch.cat((new_data_0, new_data_1, new_data_2, new_data_3, new_data_4))

        y_0 = torch.ones((num // 5)) * int(y[0])
        y_1 = torch.ones((num // 5)) * int(y[num // 5])
        y_2 = torch.ones((num // 5)) * int(y[num * 2 // 5])
        y_3 = torch.ones((num // 5)) * int(y[num * 3 // 5])
        y_4 = torch.ones((num // 5)) * int(y[num * 4 // 5])
        y = torch.cat((y_0, y_1, y_2, y_3, y_4))
        return new_data, y.reshape(-1, 1)

    def concordance_means(self, gen_data, kind1, kind2):
        alpha = concordance_cc(gen_data, kind1)
        beta = concordance_cc(gen_data, kind2)
        for i in range(len(alpha)):
            if alpha[i] >= beta[i]:
                gen_data[i] = alpha[i] * gen_data[i] + (1 - alpha[i]) * kind1[i]
            else:
                gen_data[i] = beta[i] * gen_data[i] + (1 - beta[i]) * kind2[i]
        return gen_data

    def forward(self, noise_label, data):
        label = self.label(noise_label)
        feature = self.noise(data)
        inputs = torch.cat((feature, label), dim=1)
        out = self.main(inputs.unsqueeze(1))
        out = torch.clamp(out, -0.5, 0.5)
        with torch.no_grad():
            kind_1, _ = self.candidate(data, noise_label, len(data))
            kind_2, _ = self.candidate(data, noise_label, len(data))
            kind_1 = torch.FloatTensor(kind_1)
            kind_2 = torch.FloatTensor(kind_2)
            out = self.concordance_means(out, kind_1.to(device), kind_2.to(device))
        return out, _.to(device)


# 分类器
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, 3, 2),
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, 3, 2),
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, 2),
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Flatten())

        self.fc = nn.Sequential(
            nn.Linear(480,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.conv(x)
        out = self.fc(out)
        return out




