import os
import sys
import math
import fire
from math import floor, log2
from random import random
from shutil import rmtree
import multiprocessing

import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import grad as torch_grad

import torchvision
from torchvision import transforms

from PIL import Image
from pathlib import Path

num_cores = multiprocessing.cpu_count()

# paths

CURRENT_DIR = Path('.')
RESULTS_DIR = CURRENT_DIR / 'results'
MODELS_DIR = CURRENT_DIR / 'models'

# constants

SAVE_EVERY = 10000
EXTS = ['jpg', 'png']

# helpers

def d(tensor=None):
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=d()),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-8)
    return weight * ((gradients_norm - 1) ** 2).mean()

def noise(n, latent_dim):
    return torch.randn(n, latent_dim, device=d())

def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]

def mixed_list(n, layers, latent_dim):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)

def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def image_noise(n, im_size):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).to(d())

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def leaky_relu(p):
    return nn.LeakyReLU(p, inplace=True)

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

# helper classes

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# dataset

class Dataset(data.Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.image_size = image_size        
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        if len(self.paths) == 0:
            raise Exception(f'no images found at {folder}')

        self.transform = transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# stylegan2 classes

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(emb, emb), leaky_relu(0.2)])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)
        self.conv = Conv2DMod(input_channel, 3, 1, demod=False)

        self.upsample = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x

class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel), requires_grad=True, device=d()))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdims=True) + 1e-6)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu(0.2)
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(0.2),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu(0.2)
        )

        self.downsample = nn.Conv2d(filters, filters, 3, padding = 1, stride = 2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = x + res
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        init_channels = 4 * network_capacity
        self.initial_block = nn.Parameter(torch.randn((init_channels, 4, 4), requires_grad=True, device=d()))
        filters = [init_channels] + [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]
        in_out_pairs = zip(filters[0:-1], filters[1:])

        self.blocks = nn.ModuleList([])
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size
        x = self.initial_block.expand(batch_size, -1, -1, -1)
        styles = styles.transpose(0, 1)

        rgb = None
        for style, block in zip(styles, self.blocks):
            x, rgb = block(x, rgb, style, input_noise)

        return rgb

class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16):
        super().__init__()
        num_layers = int(log2(image_size) - 1)

        blocks = []
        filters = [3] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]
        chan_in_out = list(zip(filters[0:-1], filters[1:]))

        blocks = []
        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            is_not_last = ind < (len(chan_in_out) - 1)

            block = DiscriminatorBlock(
                in_chan,
                out_chan,
                downsample = is_not_last
            )
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)
        self.to_logit = nn.Linear(2 * 2 * filters[-1], 1)

    def forward(self, x):
        b, *_ = x.shape
        x = self.blocks(x)
        x = x.reshape(b, -1)
        x = self.to_logit(x)
        return x.squeeze()

class StyleGAN2(nn.Module):
    def __init__(self, image_size, latent_dim = 512, style_depth = 8, network_capacity = 16, steps = 1, lr = 1e-4):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.99)

        self.S = StyleVectorizer(latent_dim, style_depth)
        self.G = Generator(image_size, latent_dim, network_capacity)
        self.D = Discriminator(image_size, network_capacity)

        self.SE = StyleVectorizer(latent_dim, style_depth)
        self.GE = Generator(image_size, latent_dim, network_capacity)

        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * 2, betas=(0.5, 0.9))

        self._init_weights()
        self.reset_parameter_averaging()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                m.weight.data.normal_(0., 0.02)

        for block in self.G.blocks:
            torch.nn.init.zeros_(block.to_noise1.weight)
            torch.nn.init.zeros_(block.to_noise2.weight)
            torch.nn.init.zeros_(block.to_noise1.bias)
            torch.nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x

class Trainer():
    def __init__(self, name, folder, image_size, batch_size = 4, lr = 2e-4, mixed_prob = 0.9, gradient_accumulate_every=1, *args, **kwargs):
        self.GAN = StyleGAN2(lr=lr, image_size = image_size, *args, **kwargs)
        self.GAN.to(d())

        self.folder = folder
        self.name = name

        self.batch_size = batch_size
        self.lr = lr
        self.mixed_prob = mixed_prob
        self.steps = 0

        self.av = None
        self.pl_mean = 0

        self.dataset = Dataset(folder, image_size)
        self.loader = cycle(data.DataLoader(self.dataset, num_workers = 1, batch_size = batch_size, drop_last = True, shuffle=True, pin_memory=True))
        self.gradient_accumulate_every = gradient_accumulate_every

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = 0

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

    def train(self):
        self.GAN.train()
        total_disc_loss = torch.tensor(0., device=d())
        total_gen_loss = torch.tensor(0., device=d())

        batch_size = self.batch_size

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = self.steps % 32 == 0

        # train discriminator

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()
        inputs = []

        for i in range(self.gradient_accumulate_every):
            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)
            inputs.append((style, noise))

            w_space = latent_to_w(self.GAN.S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = self.GAN.G(w_styles, noise)
            fake_output = self.GAN.D(generated_images.clone().detach())

            image_batch = next(self.loader).to(d())
            image_batch.requires_grad_()
            real_output = self.GAN.D(image_batch)

            divergence = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()
            disc_loss = divergence

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                disc_loss += gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.backward()

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.zero_grad()
        for i in range(self.gradient_accumulate_every):
            style, noise = inputs[i]
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)

            w_space = latent_to_w(self.GAN.S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = self.GAN.G(w_styles, noise)
            fake_output = self.GAN.D(generated_images)
            loss = fake_output.mean()
            gen_loss = loss

            if apply_path_penalty:
                std = 0.1 / (w_styles.std(dim = 0, keepdims = True) + 1e-8)
                w_styles_2 = w_styles + torch.randn(w_styles.shape, device=d()) / (std + 1e-8)
                pl_images = self.GAN.G(w_styles_2, noise)
                pl_lengths = ((pl_images - generated_images) ** 2).mean(dim = (1, 2, 3))
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if self.pl_mean is not None:
                    gen_loss += ((pl_lengths - self.pl_mean) ** 2).mean()

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.backward()

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.GAN.G_opt.step()

        # calculate moving averages

        if apply_path_penalty:
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)

        if self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # periodically save results

        if self.steps != 0 and self.steps % 100 == 0:
            if self.steps % 500 == 0:
                self.save(floor(self.steps / SAVE_EVERY))
            if self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(floor(self.steps / 1000))

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num = 0, trunc = 1.0):
        self.GAN.eval()
        num_rows = 8

        def generate_images(stylizer, generator, latents, noise):
            w = latent_to_w(stylizer, latents)
            w_styles = styles_def_to_tensor(w)
            generated_images = evaluate_in_chunks(self.batch_size, generator, w_styles, noise)
            generated_images.clamp_(0., 1.)
            return generated_images
    
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        # regular

        generated_images = generate_images(self.GAN.S, self.GAN.G, latents, n)
        torchvision.utils.save_image(generated_images, f'results/{self.name}/{str(num)}.jpg', nrow=num_rows)
        
        # moving averages

        generated_images = generate_images(self.GAN.SE, self.GAN.GE, latents, n)
        torchvision.utils.save_image(generated_images, f'results/{self.name}/{str(num)}-ema.jpg', nrow=num_rows)

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(d())
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = generate_images(self.GAN.SE, self.GAN.GE, mixed_latents, n)
        torchvision.utils.save_image(generated_images, f'results/{self.name}/{str(num)}-mr.jpg', nrow=num_rows)

    @torch.no_grad()
    def generate_truncated(self, style, noi, trunc = 0.5):
        latent_dim = self.GAN.G.latent_dim

        if self.av is None:
            z = noise(2000, latent_dim)
            samples = evaluate_in_chunks(self.batch_size, self.GAN.S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        num_rows = 8
        image_size = self.GAN.G.image_size
            
        w_space = []     
        for tensor, num_layers in style:
            tmp = self.GAN.S(tensor)
            av_torch = torch.from_numpy(self.av).to(d())
            tmp = trunc * (tmp - av_torch) + av_torch
            w_space.append((tmp, num_layers))

        w_styles = styles_def_to_tensor(w_space)
        generated_images = evaluate_in_chunks(self.batch_size, self.GAN.GE, w_styles, noi)
        return generated_images

    def print_log(self):
        print(f'G: {self.g_loss:.2f} | D: {self.d_loss:.2f} | GP: {self.last_gp_loss:.2f} | PL: {self.pl_mean:.2f}')

    def model_name(self, num):
        return f'models/{self.name}/model_{num}.pt'

    def init_folders(self):
        (RESULTS_DIR / self.name).mkdir(exist_ok=True)
        (MODELS_DIR / self.name).mkdir(exist_ok=True)

    def clear(self):
        rmtree(f'./models/{self.name}')
        rmtree(f'./results/{self.name}')
        self.init_folders()

    def save(self, num):
        torch.save(self.GAN.state_dict(), self.model_name(num))

    def load(self, num = -1):
        name = num
        if num == -1:
            file_paths = [p for p in Path(MODELS_DIR / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')
        self.steps = name * SAVE_EVERY
        self.GAN.load_state_dict(torch.load(self.model_name(name)))
