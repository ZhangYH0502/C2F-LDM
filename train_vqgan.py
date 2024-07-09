# -*- coding:utf-8 -*-
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torchvision.utils import save_image
import os
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

from datamodule.single_fundus_2D_datamodule import SingleFundusDatamodule
from pytorch_lightning.callbacks import ModelCheckpoint
from base.init_experiment import initExperiment

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from ldm.modules.ema import LitEma


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='vqgan_2d')
    parser.add_argument('--data_root', type=str, default='/research/deepeye/zhangyuh/data/SIGF-Total')
    parser.add_argument('--result_root', type=str, default='results/vqgan_fine')
    parser.add_argument('--image_size', default=(256, 256))
    parser.add_argument("--command", default="test")
    # train args
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--limit_train_batches", type=int, default=10000)
    parser.add_argument("--limit_val_batches", type=int, default=10000)
    parser.add_argument("--base_learning_rate", type=float, default=4.5e-6)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accumulate_grad_batches', type=int, default=4)
    parser.add_argument('--scale_lr', type=bool, default=True, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument('--num_workers', type=int, default=0)
    # lightning args
    parser.add_argument('--profiler', default='simple')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', default=[0])
    parser.add_argument('--reproduce', type=int, default=False)
    return parser


class VQModel(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        ddconfig = {
            'double_z': False,
            'z_channels': 4,
            'resolution': 512,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0
        }
        # 400 200 100 50 25
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        lossconfig = dict(disc_conditional=False,
                          disc_in_channels=3,
                          disc_num_layers=2,
                          disc_start=1,
                          disc_weight=0.6,
                          codebook_weight=1.0
                          )
        self.loss = VQLPIPSWithDiscriminator(**lossconfig)

        self.embed_dim = ddconfig["z_channels"]
        n_embed = 16384
        self.quantize = VectorQuantizer(n_embed, self.embed_dim, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)

        self.lr_g_factor = 1.0

        self.save_hyperparameters()

        self.use_ema = False
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch):
        x = batch['image']
        x_id = batch['image_id']
        return x, x_id

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = self.get_input(batch)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(
                qloss,
                x,
                xrec,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )

            self.log_dict(
                log_dict_ae,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(
                qloss,
                x,
                xrec,
                optimizer_idx,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log_dict(
                log_dict_disc,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        # with self.ema_scope():
        #     log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict
    
    def _validation_step(self, batch, batch_idx, suffix=""):
        x, _ = self.get_input(batch)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}_rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}_aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)

        # if batch_idx == 0:
        #     visuals = torch.cat([x, xrec], dim=0)
        #     self.logger.experiment.add_images(suffix+'val_visual', visuals * 0.5 + 0.5, self.current_epoch)

        return self.log_dict
        
    def test_step(self, batch, batch_idx):
        x, x_id = self.get_input(batch)
        xrec, qloss = self(x, return_pred_indices=False)
        outpath = 'results/vqgan/vqgan_2d_2023-11-28T14-36-30/lightning_logs/version_0/rec_images/'
        xrec = xrec[0, :, :, :] * 255
        xrec = xrec.permute(1, 2, 0)
        xrec[xrec > 255] = 255
        xrec[xrec < 0] = 0
        xrec = np.uint8(xrec.data.cpu().numpy())
        cv2.imwrite(os.path.join(outpath, x_id[0]), xrec)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def configure_optimizers(self):
        lr_d = self.opts.learning_rate
        lr_g = self.lr_g_factor*self.opts.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []


def main(opts):
    model = VQModel(opts)
    if opts.command == "fit":
        train_loader = SingleFundusDatamodule(
            data_root=os.path.join(opts.data_root, 'train'),
            batch_size=opts.batch_size,
            image_size=opts.image_size,
            shuffle=True,
            drop_last=True,
        )
        valid_loader = SingleFundusDatamodule(
            data_root=os.path.join(opts.data_root, 'test'),
            batch_size=1,
            image_size=opts.image_size,
            shuffle=False,
            drop_last=False,
        )
        print('train_dataset len:', len(train_loader.dataset))
        print('valid_dataset len:', len(valid_loader.dataset))
        ckpt_callback = ModelCheckpoint(
            save_last=False,
            monitor='val_rec_loss',
            mode='min',
            save_top_k=2,
            filename="model-{epoch}-{val_rec_loss}",
        )
        trainer = pl.Trainer.from_argparse_args(opts, callbacks=[ckpt_callback])
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        # trainer.validate(model, valid_loader)
    else:
        test_loader = SingleFundusDatamodule(
            data_root=os.path.join(opts.data_root, 'test'),
            batch_size=1,
            image_size=opts.image_size,
            shuffle=False,
            drop_last=False,
        )
        print('test_dataset len:', len(test_loader.dataset))
        path = 'results/vqgan/vqgan_2d_2023-11-28T14-36-30/lightning_logs/version_0/checkpoints/model-epoch=81-val_rec_loss=0.19208908081054688.ckpt'
        # model.load_from_checkpoint(path)
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
