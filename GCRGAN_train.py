import argparse
import numpy as np
import torch
from networks_2 import Generator, Discriminator, Net
# from 不加皮尔逊系数 import Generator, Discriminator
import random
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000, help="训练次数")
    parser.add_argument("--save", type=int, default=100, help="保存模型参数间隔")
    parser.add_argument("--train_dir", type=str, default='dataset/train_WT.csv', help="训练集位置")
    parser.add_argument("--gen_lr", type=float, default=0.00001, help="生成器学习率")
    parser.add_argument("--dis_lr", type=float, default=0.00001, help="辨别器学习率")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=0, help="cpu数量")

    T1 = time.time()
    # 加载参数
    opt = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 加载生成器和辨别器
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 加载优化器
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.gen_lr, betas=(opt.b1, opt.b2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=opt.dis_lr, betas=(opt.b1, opt.b2))

    # 加载分类器
    net = Net().to(device)
    optimizer_n = torch.optim.Adam(net.parameters(), lr=opt.dis_lr, betas=(opt.b1, opt.b2))
    loss_fn = torch.nn.CrossEntropyLoss()

    # 加载训练数据
    train_data = np.loadtxt(opt.train_dir, delimiter=',')[:, :1024]
    train_label1 = np.loadtxt(opt.train_dir, delimiter=',')[:, 1024]
    train_data = torch.FloatTensor(train_data).to(device)
    train_label = torch.FloatTensor(train_label1).reshape(-1, 1).to(device)

    # 加载判别器的训练数据
    test_data = np.loadtxt(opt.train_dir, delimiter=',')[:, :1024]
    test_label = np.loadtxt(opt.train_dir, delimiter=',')[:, 1024]
    test_data = torch.FloatTensor(test_data)
    test_data = torch.FloatTensor(test_data).to(device)
    test_label = torch.FloatTensor(test_label).reshape(-1, 1).to(device)

    # 开始训练
    generator.train()
    discriminator.train()
    d_loss = []
    g_loss = []
    n_loss = []
    for epoch in range(opt.epochs):

        # 训练生成器
        optimizer_g.zero_grad()
        fake_data, fake_label = generator(train_label, train_data)
        fake_scores = torch.mean(discriminator(fake_label, fake_data))
        fake_pre = net(fake_data)
        loss_pre = loss_fn(fake_pre, torch.LongTensor(train_label1).to(device))
        loss_g = -fake_scores + loss_pre
        loss_g.backward()
        optimizer_g.step()

        # 训练辨别器
        optimizer_d.zero_grad()
        fake_data, fake_label = generator(train_label, train_data)
        fake_scores = torch.mean(discriminator(fake_label, fake_data))
        real_scores = torch.mean(discriminator(test_label, test_data))
        D_lecam_real = torch.mean(torch.sqrt(torch.relu(real_scores - fake_scores) + 1e-8))
        D_lecam_fake = torch.mean(torch.sqrt(torch.relu(fake_scores - real_scores) + 1e-8))
        loss_d = - real_scores + fake_scores + 0.5 * (D_lecam_real + D_lecam_fake)
        loss_d.backward()
        optimizer_d.step()

        # 训练分类器
        optimizer_n.zero_grad()
        # pre = net(torch.cat((train_data, train_data,train_data,train_data,train_data,train_data), dim=1)[:, :1024])
        pre = net(train_data)
        loss_n = loss_fn(pre, torch.LongTensor(train_label1).to(device))
        loss_n.backward()
        optimizer_n.step()


        # 打印损失值
        print("[Epoch %d/%d] [G Loss: %f] [D Loss: %f] [N Loss: %f]" % (epoch, opt.epochs, loss_g.item(), loss_d.item(), loss_n.item()))
        d_loss.append(loss_d.cpu().detach())
        g_loss.append(loss_g.cpu().detach())
        n_loss.append(loss_n.cpu().detach())
        # 保存模型参数
        if epoch % opt.save == 0:
            torch.save(generator.state_dict(), 'results/checkpoints/gen.pth')
    T2 = time.time()
    print('程序运行时间:%s秒' % ((T2 - T1)))
    # 保存损失值
    # np.savetxt('results/loss/dis_loss.csv', d_loss, delimiter=',', fmt='%.6f')
    # np.savetxt('results/loss/gen_loss.csv', g_loss, delimiter=',', fmt='%.6f')
    # np.savetxt('results/loss/n_loss.csv', n_loss, delimiter=',', fmt='%.6f')





