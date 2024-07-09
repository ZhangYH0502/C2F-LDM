# -*- coding:utf-8 -*-
import os
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import partial
import numpy as np
import cv2
from tqdm import tqdm

import pytorch_lightning as pl
import torch
from natsort import natsorted
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image as torch_save_image

from base.init_experiment import initExperiment
from datamodule.seq_fundus_2D_datamodule_fine import SeqFundusDatamodule
from ddpm_default import DiffusionWrapper, make_beta_schedule
from ldm.lr_scheduler import LambdaLinearScheduler
from ldm.modules.diffusionmodules.util import extract_into_tensor, noise_like
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.ema import LitEma
from ldm.util import default
from train_vqgan import VQModel


def get_parser():
    parser = ArgumentParser()

    parser.add_argument("--exp_name", type=str, default='vqldm_2d_test')
    parser.add_argument('--first_stage_ckpt', type=str, default='results/vqgan/vqgan_2d_2023-11-28T14-36-30/lightning_logs/version_0/checkpoints/model-epoch=81-val_rec_loss=0.19208908081054688.ckpt')
    parser.add_argument('--data_root', type=str, default='/research/deepeye/zhangyuh/data/SIGF_Seq/')
    parser.add_argument('--result_root', type=str, default='results/vqldm_fine')
    parser.add_argument('--image_save_dir', type=str, default='results/vqldm_fine/vqldm_2d_test_2024-01-21T20-33-58/rect_images_1')

    parser.add_argument("--command", default="test")
    parser.add_argument("--image_size", default=(256, 256))
    parser.add_argument("--latent_size", default=(32, 32))
    parser.add_argument("--latent_channel", default=4)

    # train args
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--limit_train_batches", type=int, default=1000)
    parser.add_argument("--limit_val_batches", type=int, default=100)
    parser.add_argument("--base_learning_rate", type=float, default=5.0e-05)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    parser.add_argument('--scale_lr', type=bool, default=True, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument('--num_workers', type=int, default=0)

    # lightning args
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=[0])
    parser.add_argument('--reproduce', type=int, default=False)

    return parser


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class VQModelInterface(VQModel):
    def __init__(self, opts):
        super().__init__(opts)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class LDM(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.save_hyperparameters()
        unet_config = {
            'image_size': opts.latent_size,
            'in_channels': opts.latent_channel * 2,
            'out_channels': opts.latent_channel,
            'model_channels': 192,
            'attention_resolutions': [1, 2, 4, 8],
            'num_res_blocks': 2,
            'channel_mult': [1, 2, 2, 4, 4],
            'num_heads': 8,
            'use_scale_shift_norm': True,
            'resblock_updown': True
        }
        self.instantiate_first_stage(opts)
        self.model = DiffusionWrapper(unet_config, conditioning_key='fine-diff')

        self.latent_size = opts.latent_size
        self.channels = opts.latent_channel

        self.parameterization = "eps"  # all assuming fixed variance schedules
        self.loss_type = "l1"
        self.use_ema = True
        self.use_positional_encodings = False
        self.v_posterior = 0.  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        self.scale_by_std = True
        self.log_every_t = 100

        scale_factor = 1.0
        if not self.scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.register_schedule()
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def register_schedule(self,
                          given_betas=None,
                          beta_schedule="linear",
                          timesteps=1000,
                          linear_start=0.0015,
                          linear_end=0.0155,
                          cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def instantiate_first_stage(self, opts):
        model = VQModelInterface.load_from_checkpoint(opts.first_stage_ckpt)
        states = torch.load(opts.first_stage_ckpt, map_location=self.device)
        # print(states.keys(), states['state_dict'].keys())
        model.load_state_dict(states['state_dict'])
        # del model.loss
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = batch['image']
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x[:, -1, :, :, :])
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    @torch.no_grad()
    def get_input(self, batch):
        x = batch['image'].to(self.device)
        
        c1 = x[:, :6, :, :, :]
        c2 = batch['time'].to(self.device)
        c3 = batch['coarse'].to(self.device)
        
        encoder_posterior = self.encode_first_stage(x[:, -1, :, :, :])
        z = self.scale_factor * self.get_first_stage_encoding(encoder_posterior).detach()
        
        c = {'c1': c1, 'c2': c2, 'c3': c3, 'c4': z}
        
        x_id = batch['image_id']
        return z, c, x[:, -1, :, :, :], x_id

    def forward(self, x, c):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t)

    def training_step(self, batch, batch_idx):
        z, c, _, _ = self.get_input(batch)
        loss, loss_dict = self(z, c)

        if batch_idx == 0:
            self.sample_batch = batch
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def save_image(self, img, path):
        img = img[0, :, :, :] * 255
        img = img.permute(1, 2, 0)
        img[img > 255] = 255
        img[img < 0] = 0
        img = np.uint8(img.data.cpu().numpy())
        cv2.imwrite(path, img)

    def training_epoch_end(self, outputs):
        with self.ema_scope("Plotting"):
            img_save_dir = os.path.join(self.opts.default_root_dir, 'train_progress', str(self.current_epoch))
            os.makedirs(img_save_dir, exist_ok=True)
            
            z, c, x, _ = self.get_input(self.sample_batch)

            x_samples = self.sample(c=c, batch_size=1, return_intermediates=False, clip_denoised=True)
            # denoise_img_row = []
            # for x in denoise_x_row:
            #     denoise_img_row.append(self.decode_first_stage(x))
            # for img in denoise_img_row:
            #     print(img.shape)
            # self.save_image(torch.cat(denoise_img_row, dim=0), os.path.join(img_save_dir, 'gen_progressive_x0.png'))
            img_samples = self.decode_first_stage(x_samples)
            self.save_image(img_samples, os.path.join(img_save_dir, 'gen_sample.png'))

            self.save_image(x, os.path.join(img_save_dir, 'x_sample.png'))
            
            x_rec = self.decode_first_stage(z)
            self.save_image(x_rec, os.path.join(img_save_dir, 'x_rec.png'))

            # diffusion_row = []
            # for t in range(self.num_timesteps):
            #     if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
            #         t = torch.tensor([t]).repeat(z.shape[0]).to(self.device).long()
            #         noise = torch.randn_like(z)
            #         z_noisy = self.q_sample(x_start=z, t=t, noise=noise)
            #        diffusion_row.append(self.decode_first_stage(z_noisy))
            # self.save_image(torch.cat(diffusion_row, dim=0), os.path.join(img_save_dir, 'gen_forward_x0.png'))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t, cond)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = self.get_loss(model_out, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        return loss

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, c, t, clip_denoised):
        model_out = self.model(x, t, c)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised, temperature=1., noise_dropout=0., repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x0 = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0

    @torch.no_grad()
    def p_sample_loop(self, c, shape, return_intermediates=False, log_every_t=100, clip_denoised=True):
        device = self.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)
        # intermediates = [x]
        intermediates_x0 = [x]
        with tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps) as pbar:
            for i in pbar:
                x, x0 = self.p_sample(x, c, torch.full((b,), i, device=device, dtype=torch.long), clip_denoised=clip_denoised)
                if i % log_every_t == 0 or i == self.num_timesteps - 1:
                    # intermediates.append(x)
                    intermediates_x0.append(x0)
        if return_intermediates:
            return x, intermediates_x0
        return x

    @torch.no_grad()
    def sample(self, c=None, batch_size=1, return_intermediates=False, clip_denoised=True):
        return self.p_sample_loop(c,
                                  [batch_size, self.channels] + list(self.latent_size),
                                  return_intermediates=return_intermediates,
                                  clip_denoised=clip_denoised)

    def test_step(self, batch, batch_idx):
        z, c, x, x_id = self.get_input(batch)
        x_samples = self.sample(c=c, batch_size=1)
        img_samples = self.decode_first_stage(x_samples)
        
        img_save_dir = os.path.join(self.opts.image_save_dir, x_id[0])
        os.makedirs(img_save_dir, exist_ok=True)
        
        x_rec = self.decode_first_stage(z)

        self.save_image(img_samples, os.path.join(img_save_dir, 'gen.png'))
        self.save_image(x, os.path.join(img_save_dir, 'sample.png'))
        self.save_image(x_rec, os.path.join(img_save_dir, 'rec.png'))

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def configure_optimizers(self):
        lr = self.opts.learning_rate
        params = list(self.model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)

        scheduler_config = {'warm_up_steps': [10000],
                            'cycle_lengths': [10000000000000],
                            'f_start': [1e-06],
                            'f_max': [1.0],
                            'f_min': [1.0]}
        scheduler = LambdaLinearScheduler(**scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [
            {
                'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                'interval': 'step',
                'frequency': 1
            }]
        return [opt], scheduler


def main(opts):
    model = LDM(opts)
    if opts.command == "fit":
        train_loader = SeqFundusDatamodule(
            data_root=opts.data_root,
            batch_size=opts.batch_size,
            image_size=opts.image_size,
            shuffle=True,
            drop_last=True,
            mode='train',
        )
        ckpt_callback = ModelCheckpoint(
            save_last=False,
            save_top_k=-1,
            filename="model-{epoch}",
        )
        trainer = pl.Trainer.from_argparse_args(opts, callbacks=[ckpt_callback])
        trainer.fit(model, train_loader)
    else:
        test_loader = SeqFundusDatamodule(
            data_root=opts.data_root,
            batch_size=opts.batch_size,
            image_size=opts.image_size,
            shuffle=False,
            drop_last=False,
            mode='train',
        )
        path = 'results/vqldm_fine/vqldm_2d_test_2024-01-21T20-33-58/lightning_logs/version_0/checkpoints/model-epoch=199.ckpt'
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        trainer = pl.Trainer.from_argparse_args(opts)
        trainer.test(model, test_loader)


if __name__ == '__main__':
    parser = get_parser()
    opts = parser.parse_args()
    os.environ['TORCH_HOME'] = '/research/deepeye/zhangyuh/seqpred_1/pre-trained'
    print("Using", torch.cuda.device_count(), "GPUs!")
    initExperiment(opts)
    main(opts)
