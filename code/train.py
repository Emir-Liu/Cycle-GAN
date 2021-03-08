import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator,Discriminator
from utils import ReplayBuffer,LambdaLR,Logger,weights_init_normal
from util import save_model
from datasets import ImageDataset
from Config import Config

# 下面是配置相关的参数,修改了一下，之前的太麻烦
opt = Config()

###### Definition of variables ######
# 建立网络
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)


if opt.epoch == 0:
    # 网络的初始化
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)
else:
    chkpt = torch.load(opt.output_path+'netG_A2B'+'.ep%d'%opt.epoch)
    netG_A2B.load_state_dict(chkpt)
    
    chkpt = torch.load(opt.output_path+'netG_B2A'+'.ep%d'%opt.epoch)
    netG_B2A.load_state_dict(chkpt)

    chkpt = torch.load(opt.output_path+'netD_A'+'.ep%d'%opt.epoch)
    netD_A.load_state_dict(chkpt)

    chkpt = torch.load(opt.output_path+'netD_B'+'.ep%d'%opt.epoch)
    netD_B.load_state_dict(chkpt)

# 是否使用GPU
if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()


# 建立损失函数：GAN、Cycle和identity
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# 优化器和学习速率设置:G,D_A,D_B
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# 输入和目标的内存配置
# A和B的输入，实际目标和虚假目标
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

# ？？buffer
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
'''
数据预处理
transformers.Resize(size,interpolation=2) 按照比例将图片进行放缩处理,如果size为一个数字，呢么就是最短的边和它相匹配

transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
在随意位置裁剪给定图像，size为int时候，则输出为一个方形

transforms.RandomHorizontalFlip(p = 0.5)
以给定的概率水平翻转给定的图像

transforms.ToTensor()
转换为tensor格式

transforms.Normalize(mean, std, inplace=False)
对图片进行归一化处理，每个维度表示一个channel
'''
transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC),
                transforms.RandomCrop(opt.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

# 调用标准的dataloader,可惜dataset是自己实现的
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# 绘制误差曲线,待定
logger = Logger(opt.n_epochs, len(dataloader))

# 训练部分
'''
终于明白了为什么会有opt.epoch，因为有的时候，训练的时候会有中断，为了继续训练所以设置了它
'''
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # 输入部分，这部分有些地方可以修改关于Variable
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        # Generators A2B and B2A 首先是Generator部分
        optimizer_G.zero_grad()

        # Identity loss
        # A -> G_A2B -> A 通过这个转换，减少前后两个的差
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*5.0

        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*5.0

        # GAN loss
        # A -> G_A2B -> D_B 判别器进行判别，减少判别器误差
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        # A -> G_A2B -> G_B2A 减少输入和输出的误差
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    save_model(netG_A2B,epoch,opt.output_path,'netG_A2B')
    save_model(netG_B2A, epoch, opt.output_path, 'netG_B2A')
    save_model(netD_A, epoch, opt.output_path, 'netD_A')
    save_model(netD_B, epoch, opt.output_path, 'netD_B')

    torch.save(netG_A2B.state_dict(), '../../output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), '../../output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), '../../output/netD_A.pth')
    torch.save(netD_B.state_dict(), '../../output/netD_B.pth')
###################################
