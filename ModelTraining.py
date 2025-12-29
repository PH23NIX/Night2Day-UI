# ModelTraining.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import itertools

# ==================== EDITABLE PARAMETERS ====================
PARAMS = {
    # Directories
    'day_images_dir': r'D:\Program Files (x86)\Night2Day UI\Data\CloseVehicle\Day',
    'night_images_dir': r'D:\Program Files (x86)\Night2Day UI\Data\CloseVehicle\Night\Train',
    'checkpoint_dir': r'D:\Program Files (x86)\Night2Day UI\Models\checkpoints',
    'final_dir': r'D:\Program Files (x86)\Night2Day UI\Models\final',
    'output_dir': r'D:\Program Files (x86)\Night2Day UI\Output\close_gen_day',

    # Resume training
    'resume_checkpoint': None,  # Path to checkpoint to resume from, or None to start fresh

    # Training parameters
    'epochs': 100,
    'batch_size': 4,
    'learning_rate_G': 0.0001,
    'learning_rate_D': 0.00005,
    'beta1': 0.5,
    'beta2': 0.999,
    'gradient_clip_value': 0.5,  # Gradient clipping for generators
    'label_smoothing': 0.1,

    # Loss weights
    'lambda_cycle': 10.0,
    'lambda_identity': 5,

    # Model parameters
    'img_size': 512,
    'num_residual_blocks': 9,
    'ngf': 64,
    'ndf': 64,

    # Training settings
    'save_interval': 5,
    'sample_interval': 100,  # Generate sample image
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
# ============================================================

# ------------------ MODEL DEFINITIONS ------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_channels=3, num_residual_blocks=9, ngf=64):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, input_channels, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, ndf=64):
        super().__init__()
        def disc_block(in_f, out_f, norm=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *disc_block(input_channels, ndf, norm=False),
            *disc_block(ndf, ndf*2),
            *disc_block(ndf*2, ndf*4),
            *disc_block(ndf*4, ndf*8),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(ndf*8, 1, 4, padding=1)
        )
    def forward(self, x):
        return self.model(x)

# ------------------ DATASET ------------------
class ImageDataset(Dataset):
    def __init__(self, day_dir, night_dir, transform=None):
        self.day_dir = day_dir
        self.night_dir = night_dir
        self.transform = transform
        self.day_images = sorted([f for f in os.listdir(day_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.night_images = sorted([f for f in os.listdir(night_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.day_len = len(self.day_images)
        self.night_len = len(self.night_images)
        self.length = max(self.day_len, self.night_len)
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        day_img = Image.open(os.path.join(self.day_dir, self.day_images[idx % self.day_len])).convert('RGB')
        night_img = Image.open(os.path.join(self.night_dir, self.night_images[idx % self.night_len])).convert('RGB')
        if self.transform:
            day_img = self.transform(day_img)
            night_img = self.transform(night_img)
        return {'day': day_img, 'night': night_img}

# ------------------ UTILITIES ------------------
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []
    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if np.random.uniform() > 0.5:
                    i = np.random.randint(0, self.max_size)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def plot_losses(loss_history, epoch, step, save_path, log_fn=None):
    fig, ax = plt.subplots(figsize=(12,7))
    for key in loss_history.keys():
        ax.plot(loss_history[key], label=key, linewidth=2)
    ax.set_xlabel('Steps (x100)')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Losses - Epoch {epoch+1}, Step {step}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    if log_fn:
        log_fn(f"\nEpoch {epoch+1}, Step {step} Losses:")
        for key, values in loss_history.items():
            if values:
                log_fn(f"{key}: {values[-1]:.4f}")

# ------------------ TRAINING ------------------
def train_model(log_fn=None, progress_fn=None):
    # Directories
    os.makedirs(PARAMS['checkpoint_dir'], exist_ok=True)
    os.makedirs(PARAMS['final_dir'], exist_ok=True)
    os.makedirs(PARAMS['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(PARAMS['output_dir'], 'samples'), exist_ok=True)
    os.makedirs(os.path.join(PARAMS['output_dir'], 'loss_plots'), exist_ok=True)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((PARAMS['img_size'], PARAMS['img_size']), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    dataset = ImageDataset(PARAMS['day_images_dir'], PARAMS['night_images_dir'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=PARAMS['batch_size'], shuffle=True, num_workers=PARAMS['num_workers'])

    if log_fn:
        log_fn(f"Dataset loaded: {len(dataset)} image pairs")

    # Models
    G_night2day = Generator(num_residual_blocks=PARAMS['num_residual_blocks'], ngf=PARAMS['ngf']).to(PARAMS['device'])
    G_day2night = Generator(num_residual_blocks=PARAMS['num_residual_blocks'], ngf=PARAMS['ngf']).to(PARAMS['device'])
    D_day = Discriminator(ndf=PARAMS['ndf']).to(PARAMS['device'])
    D_night = Discriminator(ndf=PARAMS['ndf']).to(PARAMS['device'])

    G_night2day.apply(weights_init_normal)
    G_day2night.apply(weights_init_normal)
    D_day.apply(weights_init_normal)
    D_night.apply(weights_init_normal)

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = optim.Adam(itertools.chain(G_night2day.parameters(), G_day2night.parameters()), lr=PARAMS['learning_rate_G'], betas=(PARAMS['beta1'], PARAMS['beta2']))
    optimizer_D_day = optim.Adam(D_day.parameters(), lr=PARAMS['learning_rate_D'], betas=(PARAMS['beta1'], PARAMS['beta2']))
    optimizer_D_night = optim.Adam(D_night.parameters(), lr=PARAMS['learning_rate_D'], betas=(PARAMS['beta1'], PARAMS['beta2']))

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch-PARAMS['epochs']//2)/(PARAMS['epochs']//2))
    lr_scheduler_D_day = optim.lr_scheduler.LambdaLR(optimizer_D_day, lr_lambda=lambda epoch: 1.0 - max(0, epoch-PARAMS['epochs']//2)/(PARAMS['epochs']//2))
    lr_scheduler_D_night = optim.lr_scheduler.LambdaLR(optimizer_D_night, lr_lambda=lambda epoch: 1.0 - max(0, epoch-PARAMS['epochs']//2)/(PARAMS['epochs']//2))

    fake_day_buffer = ReplayBuffer()
    fake_night_buffer = ReplayBuffer()

    loss_history = {'loss_G': [], 'loss_D': [], 'loss_cycle': [], 'loss_identity': [], 'loss_GAN': []}
    global_step = 0
    start_epoch = 0

    # ------------------ RESUME CHECKPOINT ------------------
    if PARAMS['resume_checkpoint'] is not None and os.path.exists(PARAMS['resume_checkpoint']):
        checkpoint = torch.load(PARAMS['resume_checkpoint'], map_location=PARAMS['device'])
        G_night2day.load_state_dict(checkpoint['G_night2day'])
        G_day2night.load_state_dict(checkpoint['G_day2night'])
        D_day.load_state_dict(checkpoint['D_day'])
        D_night.load_state_dict(checkpoint['D_night'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D_day.load_state_dict(checkpoint['optimizer_D_day'])
        optimizer_D_night.load_state_dict(checkpoint['optimizer_D_night'])
        if 'lr_scheduler_G' in checkpoint:
            lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
            lr_scheduler_D_day.load_state_dict(checkpoint['lr_scheduler_D_day'])
            lr_scheduler_D_night.load_state_dict(checkpoint['lr_scheduler_D_night'])
        if 'loss_history' in checkpoint:
            loss_history = checkpoint['loss_history']
            global_step = len(loss_history['loss_G'])
        start_epoch = checkpoint['epoch'] + 1
        if log_fn:
            log_fn(f"Resumed from epoch {start_epoch}, global_step {global_step}")

    # ------------------ TRAINING LOOP ------------------
    for epoch in range(start_epoch, PARAMS['epochs']):
        for i, batch in enumerate(dataloader):
            real_day = batch['day'].to(PARAMS['device'])
            real_night = batch['night'].to(PARAMS['device'])

            disc_output_size = PARAMS['img_size'] // 16
            smooth = PARAMS['label_smoothing']
            valid = torch.ones((real_day.size(0), 1, disc_output_size, disc_output_size), device=PARAMS['device']) * (1.0 - smooth)
            fake = torch.zeros((real_day.size(0), 1, disc_output_size, disc_output_size), device=PARAMS['device']) + smooth*0.5

            # Generators
            optimizer_G.zero_grad()
            loss_id_day = criterion_identity(G_night2day(real_day), real_day)
            loss_id_night = criterion_identity(G_day2night(real_night), real_night)
            loss_identity = (loss_id_day + loss_id_night)/2
            fake_day = G_night2day(real_night)
            loss_GAN_n2d = criterion_GAN(D_day(fake_day), valid)
            fake_night = G_day2night(real_day)
            loss_GAN_d2n = criterion_GAN(D_night(fake_night), valid)
            loss_GAN = (loss_GAN_n2d + loss_GAN_d2n)/2
            recov_day = G_night2day(fake_night)
            recov_night = G_day2night(fake_day)
            loss_cycle = (criterion_cycle(recov_day, real_day) + criterion_cycle(recov_night, real_night))/2
            loss_G = loss_GAN + PARAMS['lambda_cycle']*loss_cycle + PARAMS['lambda_identity']*loss_identity
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(G_night2day.parameters(), PARAMS['gradient_clip_value'])
            torch.nn.utils.clip_grad_norm_(G_day2night.parameters(), PARAMS['gradient_clip_value'])
            optimizer_G.step()

            # Discriminator Day
            optimizer_D_day.zero_grad()
            loss_real = criterion_GAN(D_day(real_day), valid)
            loss_fake = criterion_GAN(D_day(fake_day_buffer.push_and_pop(fake_day).detach()), fake)
            loss_D_day = (loss_real + loss_fake)/2
            loss_D_day.backward()
            optimizer_D_day.step()

            # Discriminator Night
            optimizer_D_night.zero_grad()
            loss_real = criterion_GAN(D_night(real_night), valid)
            loss_fake = criterion_GAN(D_night(fake_night_buffer.push_and_pop(fake_night).detach()), fake)
            loss_D_night = (loss_real + loss_fake)/2
            loss_D_night.backward()
            optimizer_D_night.step()

            loss_D = (loss_D_day + loss_D_night)/2

            # Update loss history
            loss_history['loss_G'].append(loss_G.item())
            loss_history['loss_D'].append(loss_D.item())
            loss_history['loss_cycle'].append(loss_cycle.item())
            loss_history['loss_identity'].append(loss_identity.item())
            loss_history['loss_GAN'].append(loss_GAN.item())

            # Update progress bar
            if progress_fn:
                progress_fn(global_step, PARAMS['epochs']*len(dataloader))

            global_step += 1

            # Save samples
            if global_step % PARAMS['sample_interval'] == 0:
                sample_night = real_night[:1]
                sample_day_real = real_day[:1]
                sample_day_fake = G_night2day(sample_night)
                sample_save_path = os.path.join(PARAMS['output_dir'], 'samples', f'epoch{epoch+1}_step{global_step}.png')
                # Convert to image
                output_img = (sample_day_fake.cpu().squeeze().permute(1,2,0).numpy()*0.5+0.5).clip(0,1)
                plt.imsave(sample_save_path, output_img)
                if log_fn:
                    log_fn(f"Saved sample: {sample_save_path}")

        # Plot losses
        loss_plot_path = os.path.join(PARAMS['output_dir'], 'loss_plots', f'loss_epoch_{epoch+1}.png')
        plot_losses(loss_history, epoch, global_step, loss_plot_path, log_fn=log_fn)

        # Update schedulers
        lr_scheduler_G.step()
        lr_scheduler_D_day.step()
        lr_scheduler_D_night.step()

        # Save checkpoint
        if (epoch+1) % PARAMS['save_interval'] == 0:
            checkpoint_path = os.path.join(PARAMS['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'G_night2day': G_night2day.state_dict(),
                'G_day2night': G_day2night.state_dict(),
                'D_day': D_day.state_dict(),
                'D_night': D_night.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D_day': optimizer_D_day.state_dict(),
                'optimizer_D_night': optimizer_D_night.state_dict(),
                'lr_scheduler_G': lr_scheduler_G.state_dict(),
                'lr_scheduler_D_day': lr_scheduler_D_day.state_dict(),
                'lr_scheduler_D_night': lr_scheduler_D_night.state_dict(),
                'loss_history': loss_history
            }, checkpoint_path)
            if log_fn:
                log_fn(f"Saved checkpoint: {checkpoint_path}")

    # Save final models
    torch.save(G_night2day.state_dict(), os.path.join(PARAMS['final_dir'], 'night2day_final.pth'))
    torch.save(G_day2night.state_dict(), os.path.join(PARAMS['final_dir'], 'day2night_final.pth'))
    if log_fn:
        log_fn(f"\n✅ Training finished. Final models saved to {PARAMS['final_dir']}")

# ------------------ IMAGE CONVERSION ------------------
def convert_images(model_path, input_dir, output_dir, log_fn=None, progress_fn=None):
    device = PARAMS['device']
    os.makedirs(output_dir, exist_ok=True)

    G = Generator(num_residual_blocks=PARAMS['num_residual_blocks'], ngf=PARAMS['ngf']).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    transform = transforms.Compose([
        transforms.Resize((PARAMS['img_size'], PARAMS['img_size']), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    total = len(images)
    for idx, img_name in enumerate(images):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            fake_img = G(img_tensor)
        output_img = ((fake_img.squeeze().permute(1,2,0).cpu().numpy()*0.5)+0.5).clip(0,1)
        plt.imsave(os.path.join(output_dir, img_name), output_img)
        if log_fn:
            log_fn(f"Converted: {img_name}")
        if progress_fn:
            progress_fn(idx+1, total)

    if log_fn:
        log_fn(f"\n✅ Conversion finished. Output saved to {output_dir}")
