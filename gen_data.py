from networks_2 import Generator, Discriminator
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_gen", type=int, default=19, help="生成数据倍数")

opt = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data = np.loadtxt('dataset/train_WT.csv', delimiter=',')
train_data = np.loadtxt('dataset/train_WT.csv', delimiter=',')[:, :1024]
train_label = np.loadtxt('dataset/train_WT.csv', delimiter=',')[:, 1024]
train_data = torch.FloatTensor(train_data).to(device)
train_label = torch.FloatTensor(train_label).reshape(-1, 1).to(device)

new_data = np.zeros((1, 1024))
new_label = np.zeros(1)
generator = Generator().to(device)
gen = torch.load('results/checkpoints/gen.pth')
generator.load_state_dict(gen)
generator.eval()
count = 0
for i in range(opt.num_gen):
    gen_data, gen_label = generator(train_label, train_data)
    gen_data = torch.squeeze(gen_data)
    gen_data = gen_data.cpu().detach().numpy()
    gen_label = torch.squeeze(gen_label)
    gen_label = gen_label.cpu().detach().numpy()
    new_data = np.concatenate((new_data, gen_data))
    new_label = np.concatenate((new_label, gen_label))
    torch.cuda.empty_cache()
    print(count)
    count = count + 1

new_data = np.delete(new_data, 0, axis=0)
new_label = np.delete(new_label, 0, axis=0)
new_label = new_label.reshape(-1, 1)
out = np.concatenate((new_data, new_label), axis=1)
out_ = np.concatenate((data, out), axis=0)
print(out_.shape)
np.savetxt('results/gen_data/gen.csv', out_, delimiter=',')



