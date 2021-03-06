
import argparse
import os
import itertools
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable
import natsort
from models import *
from images_to_video import *
from datasets import *
from utils import *
from PIL import Image

import torch
# 65,35.25,15
for loop in range(5, 150, 5):
    print(loop)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=loop, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    # parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
    parser.add_argument("--dataset_name", type=str, default="day2night_cycle_1", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=3, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=(256), help="size of image height")
    parser.add_argument("--img_width", type=int, default=(256), help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=1, help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
    opt = parser.parse_args()
    print(opt)

    # Create sample and checkpoint directories
    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    cuda = torch.cuda.is_available()

    input_shape = (opt.channels, opt.img_height, opt.img_width)

    # Initialize generator and discriminator
    G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
    G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
    D_A = Discriminator(input_shape)
    D_B = Discriminator(input_shape)

    if cuda:
        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()
    if opt.epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load("/home/mayank_s/codebase/others/gans/PyTorch-GAN/implementations/cyclegan/saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
        G_BA.load_state_dict(torch.load("/home/mayank_s/codebase/others/gans/PyTorch-GAN/implementations/cyclegan/saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
        D_A.load_state_dict(torch.load("/home/mayank_s/codebase/others/gans/PyTorch-GAN/implementations/cyclegan/saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
        D_B.load_state_dict(torch.load("/home/mayank_s/codebase/others/gans/PyTorch-GAN/implementations/cyclegan/saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # input_folder='/home/mayank_s/Desktop/template/farm_2/farm_2_images2_scaled'
    input_folder='/home/mayank_s/Desktop/ppt/london_timelapse_720p_img'
    img_saved='/home/mayank_s/codebase/others/gans/gan_output'
    video_save_path='/home/mayank_s/codebase/others/gans/gan_video'
    for root, _, filenames in os.walk(input_folder):
        if (len(filenames) == 0):
            print("Input folder is empty")
            # return 1
        # time_start = time.time()
        filenames = natsort.natsorted(filenames, reverse=False)

        for i,filename in enumerate(filenames):
            print(filename)
            file_path = (os.path.join(root, filename));
            # ################################33
            img0 = Image.open(file_path)
            img0 = img0.convert("RGB")
            # transform = transforms.Compose([transforms.Resize((30, 30)), transforms.ToTensor(),normalize])
            # transforms_ = transforms.Compose([
            #     transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
            #     transforms.RandomCrop((opt.img_height, opt.img_width)),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # ])
            transforms_ = transforms.Compose([
                transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            img = transforms_(img0)
            img=img.unsqueeze(0)

            G_AB.eval()
            G_BA.eval()
            real_A = Variable(img.type(Tensor))
            # real_A = Variable(img["A"].type(Tensor))
            fake_B = G_AB(real_A)
            real_A = make_grid(real_A, nrow=1, normalize=True)
            # real_B = make_grid(real_B, nrow=5, normalize=True)
            # fake_A = make_grid(fake_A, nrow=5, normalize=True)
            fake_B = make_grid(fake_B, nrow=1, normalize=True)
            # Arange images along y-axis
            image_grid = torch.cat((real_A, fake_B), 1)
            # save_image(image_grid, "/home/mayank_s/codebase/others/gans/gan_output/%s.png" % str(i), normalize=False)
            final_image_save=img_saved+"/"+filename
            save_image(image_grid, final_image_save, normalize=False)
            # save_image(image_grid, "/home/mayank_s/codebase/others/gans/gan_output/%s" % filename, normalize=False)

    output_path=(video_save_path+ "/%s_%s.avi" % (opt.dataset_name,opt.epoch))
    print(output_path)
    # output_path=video_save_path/%s/%s.png" % (opt.dataset_name, batches_done)
    make_video(img_saved,outvid=output_path)