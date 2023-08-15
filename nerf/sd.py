from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from torch.cuda.amp import custom_bwd, custom_fwd
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None



class StableDiffusion(nn.Module):
    def __init__(self, opt, device, sd_path, sd_version='2.0'):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.opt = opt
        self.device = device
        self.sd_version = sd_version

        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * self.opt.sd_max_step)

        print(f'[INFO] loading stable diffusion...')

        self.vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet").to(self.device)

        if is_xformers_available():
            print('*'*100)
            print('enable xformers')
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )

        self.scheduler = DDIMScheduler.from_config(sd_path, subfolder="scheduler")

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for conve

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def train_step(self, text_embeddings, pred_rgb, ratio=0, guidance_scale=100, lamb = 1.0):
        
        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        if self.opt.sd_img_size == 512:
            pred_rgb = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        else:
            pred_rgb = F.interpolate(pred_rgb, (768, 768), mode='bilinear', align_corners=False)

        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level

        if ratio > 0:
            # min_s = max(self.min_step, int((1-ratio-0.05)*self.num_train_timesteps))
            max_s = int((0.25 + 0.75 *(1-ratio)) * self.opt.sd_max_step * self.num_train_timesteps)
            t = torch.randint(self.min_step, max_s + 1, [1], dtype=torch.long, device=self.device)

        else:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        latents = self.encode_imgs(pred_rgb)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])

        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)
        grad = lamb * grad.clamp(-1, 1)
        # grad = torch.nan_to_num(grad)

        loss = SpecifyGradient.apply(latents, grad)

        return loss


    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents




