import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from vqvae2 import VQVAE, VQVAE2

class Encoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_residual_layers, residual_dim, resample_factors, use_batch_norm, use_rezero):
        super().__init__()
        modules = []
        current_dim = in_dim
        for factor in resample_factors:
            modules.append(torch.nn.Conv2d(current_dim, hidden_dim, 4, stride=factor, padding=1))
            modules.append(torch.nn.ReLU())
            current_dim = hidden_dim
        for _ in range(num_residual_layers):
            modules.append(ResidualBlock(hidden_dim, residual_dim, use_batch_norm, use_rezero))
        self.net = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

class Decoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_residual_layers, residual_dim, resample_factors, use_batch_norm, use_rezero):
        super().__init__()
        modules = []
        current_dim = in_dim
        for _ in range(num_residual_layers):
            modules.append(ResidualBlock(current_dim, residual_dim, use_batch_norm, use_rezero))
        for factor in resample_factors:
            modules.append(torch.nn.ConvTranspose2d(current_dim, hidden_dim, 4, stride=factor, padding=1))
            modules.append(torch.nn.ReLU())
            current_dim = hidden_dim
        modules.append(torch.nn.Conv2d(hidden_dim, 3, 3, stride=1, padding=1)) # Output channels hardcoded to 3
        self.net = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, residual_dim, use_batch_norm, use_rezero):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(in_dim) if use_batch_norm else torch.nn.Identity()
        self.conv1 = torch.nn.Conv2d(in_dim, residual_dim, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(residual_dim, in_dim, 3, stride=1, padding=1)
        self.rezero = ReZero(in_dim) if use_rezero else torch.nn.Identity()

    def forward(self, x):
        h = self.norm(x)
        h = self.relu(h)
        h = self.conv1(h)
        h = self.relu(h)
        h = self.conv2(h)
        return x + self.rezero(h)

class ReZero(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        return self.alpha * x

class VectorQuantizer(torch.nn.Module):
    def __init__(self, n_e, e_dim, beta, gumbel_temperature=None, init_type='uniform', cosine=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.gumbel_temperature = gumbel_temperature
        self.embedding = torch.nn.Embedding(self.n_e, self.e_dim)
        if init_type == 'uniform':
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        elif init_type == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(self.embedding.weight, a=0, mode='fan_in', nonlinearity='relu')
        elif init_type == 'normal':
            self.embedding.weight.data.normal_(0, 1)
        if cosine:
            self.embedding.weight.data = torch.nn.functional.normalize(self.embedding.weight.data, dim=1)

        self.loss = None

    def forward(self, z):
        z = z.contiguous()
        flat_z = z.view(-1, self.e_dim)
        dist = torch.cdist(flat_z, self.embedding.weight)

        if self.gumbel_temperature is not None and self.training:
            # Gumbel-softmax trick
            temp = self.gumbel_temperature
            logits = -dist
            prob = torch.nn.functional.softmax(logits / temp, dim=-1)
            sample = torch.nn.functional.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
            embed = torch.matmul(sample, self.embedding.weight)
            embed = embed.view_as(z)
            self.loss = torch.mean(torch.sum(-prob * torch.log_softmax(logits / temp, dim=-1), dim=-1)) * self.beta
            return embed, prob, sample.argmax(dim=-1).view_as(z[:, 0, ...])
        else:
            encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
            sample = self.embedding(encoding_indices).view_as(z)
            encoding_one_hot = torch.zeros(encoding_indices.size(0), self.n_e, device=z.device)
            encoding_one_hot.scatter_(1, encoding_indices, 1)
            quantize = torch.matmul(encoding_one_hot, self.embedding.weight).view_as(z)
            self.loss = torch.mean((quantize.detach() - z)**2) + self.beta * torch.mean((quantize - z.detach())**2)
            return quantize, encoding_one_hot, encoding_indices.view_as(z[:, 0, ...])

def load_model(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(image_path, image_size):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def postprocess_image(recon_tensor):
    recon_img = recon_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    recon_img = recon_img * 255
    recon_img = np.clip(recon_img, 0, 255).astype(np.uint8)
    return recon_img

def main():


    # --- Configuration ---
    checkpoint_path = "/workspace/exp/vqvae_2025-06-28_20-03-07/checkpoints/state_dict_final.pt"
    image_path = "/workspace/vqvae-2/data/colo/kusuhara-2.png"
    image_size = (256, 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build Model ---
    config = {
        "in_dim": 3,
        "hidden_dim": 128,
        "codebook_dim": 64,
        "codebook_size": 512,
        "num_residual_layers": 2,
        "residual_dim": 64,
        "resample_factors": [2, 2, 2, 2]
    }
    model = VQVAE2.build_from_config(config)
    model.to(device)

    # --- Load Model ---
    model = load_model(checkpoint_path, model)

    # --- Preprocess Image ---
    input_tensor = preprocess_image(image_path, image_size).to(device)

    # --- Inference ---
    with torch.no_grad():
        reconstruction, _, _, _ = model(input_tensor)

    # --- Postprocess Image ---
    original_img = Image.open(image_path).resize(image_size)
    reconstructed_img_np = postprocess_image(reconstruction)
    reconstructed_img = Image.fromarray(reconstructed_img_np)

    # --- Display Results ---
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img)
    plt.title("Reconstructed Image")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("/workspace/inhouse-vqvae/vqvae2/result/kusu2222.png")

if __name__ == "__main__":
    main()