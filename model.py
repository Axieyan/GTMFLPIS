from torchvision import models
from CNN_text import ResNet_text_50
import transformers as ppb
from torch.nn import init
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from transformers import BertModel, RobertaModel
from CMPM import Loss
import torch.optim as optim
from test_utils import test_map
from scheduler import GradualWarmupScheduler # https://github.com/ildoonet/pytorch-gradual-warmup-lr

class ResNet_image_50(nn.Module):
    def __init__(self):
        super(ResNet_image_50, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        resnet50.layer4[0].downsample[0].stride = (1, 1)
        resnet50.layer4[0].conv2.stride = (1, 1)
        self.base1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,  # 256 64 32
        )
        self.base2 = nn.Sequential(
            resnet50.layer2,  # 512 32 16
        )
        self.base3 = nn.Sequential(
            resnet50.layer3,  # 1024 16 8
        )
        self.base4 = nn.Sequential(
            resnet50.layer4  # 2048 16 8
        )

    def forward(self, x):
        x1 = self.base1(x)
        x2 = self.base2(x1)
        x3 = self.base3(x2)
        x4 = self.base4(x3)
        return x1, x2, x3, x4

class TIPCB(pl.LightningModule):
    def __init__(self, args, val_len = None):
        super(TIPCB, self).__init__()
        self.args = args

        self.model_img = ResNet_image_50()
        self.model_txt = ResNet_text_50(args)

        self.compute_loss = Loss(args)

        if args.language == "en":
            self.text_embed = RobertaModel.from_pretrained("/aicity/finetune_LM_WCB/models/en-roberta")
        elif args.language == "th":
            self.text_embed = RobertaModel.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
            # self.text_embed = RobertaModel.from_pretrained("/aicity/finetune_LM_WCB/models/th2")
        # self.text_embed.train()
        self.text_embed.eval()
        # self.BERT = True
        for p in self.text_embed.parameters():
            # p.requires_grad = True
            p.requires_grad = False

        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        if val_len: # length of the validation dataloader
            self.register_buffer("sigma", torch.eye(3))
            # you can now access self.sigma anywhere in your module
            max_size = args.batch_size * val_len
            self.register_buffer("images_bank", torch.zeros((max_size, args.feature_size)))
            self.register_buffer("text_bank",torch.zeros((max_size, args.feature_size)))
            self.register_buffer("labels_bank",torch.zeros(max_size))
            self.index = 0
            self.max_size = max_size
            self.feature_size = args.feature_size

    def _forward(self, img, txt, mask, training=False):
        with torch.no_grad():
            txt = self.text_embed(txt, attention_mask=mask)
            txt = txt[0] # (batch_size, sequence_length, hidden_size)
            assert txt.shape[2] == 768, "shape is wrong"
            txt = txt.unsqueeze(1) # Bx1xLxH
            txt = txt.permute(0, 3, 1, 2) # BxHx1xL

        _, _, img3, img4 = self.model_img(img)  # img4: batch x 2048 x 24 x 8
        img_f3 = self.max_pool(img3).squeeze(dim=-1).squeeze(dim=-1)
        img_f41 = self.max_pool(img4[:, :, 0:4, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f42 = self.max_pool(img4[:, :, 4:8, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f43 = self.max_pool(img4[:, :, 8:12, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f44 = self.max_pool(img4[:, :, 12:16, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f45 = self.max_pool(img4[:, :, 16:20, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f46 = self.max_pool(img4[:, :, 20:, :]).squeeze(dim=-1).squeeze(dim=-1)
        img_f4 = self.max_pool(img4).squeeze(dim=-1).squeeze(dim=-1)

        txt3, txt41, txt42, txt43, txt44, txt45, txt46 = self.model_txt(txt)  # txt4: batch x 2048 x 1 x 64
        txt_f3 = self.max_pool(txt3).squeeze(dim=-1).squeeze(dim=-1)
        txt_f41 = self.max_pool(txt41)
        txt_f42 = self.max_pool(txt42)
        txt_f43 = self.max_pool(txt43)
        txt_f44 = self.max_pool(txt44)
        txt_f45 = self.max_pool(txt45)
        txt_f46 = self.max_pool(txt46)
        txt_f4 = self.max_pool(torch.cat([txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46], dim=2)).squeeze(dim=-1).squeeze(dim=-1)
        txt_f41 = txt_f41.squeeze(dim=-1).squeeze(dim=-1)
        txt_f42 = txt_f42.squeeze(dim=-1).squeeze(dim=-1)
        txt_f43 = txt_f43.squeeze(dim=-1).squeeze(dim=-1)
        txt_f44 = txt_f44.squeeze(dim=-1).squeeze(dim=-1)
        txt_f45 = txt_f45.squeeze(dim=-1).squeeze(dim=-1)
        txt_f46 = txt_f46.squeeze(dim=-1).squeeze(dim=-1)
    
        if training:
            return img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, \
                    txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46
        else:
            return img_f4, txt_f4

    def training_step(self, batch, idx):
        img, txt, labels, mask = batch
        img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, \
                    txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46 = self._forward(img, txt, mask, training=True)

        loss = self.compute_loss(
            img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46,
            txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, labels)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, idx):
        img, txt, labels, mask = batch
        img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46, \
                    txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46 = self._forward(img, txt, mask, training=True)

        interval = int(img.shape[0])

        loss = self.compute_loss(
            img_f3, img_f4, img_f41, img_f42, img_f43, img_f44, img_f45, img_f46,
            txt_f3, txt_f4, txt_f41, txt_f42, txt_f43, txt_f44, txt_f45, txt_f46, labels)

        self.images_bank[self.index: self.index + interval] = img_f4
        self.text_bank[self.index: self.index + interval] = txt_f4
        self.labels_bank[self.index:self.index + interval] = labels
        self.index += interval
        
        self.log('val_loss', loss)

    def on_validation_epoch_start(self):
        self.images_bank = torch.zeros((self.max_size, self.feature_size))
        self.text_bank = torch.zeros((self.max_size, self.feature_size))
        self.labels_bank = torch.zeros(self.max_size)
        self.index = 0

    def on_validation_epoch_end(self):
        self.images_bank = self.images_bank[:self.index]
        self.text_bank = self.text_bank[:self.index]
        self.labels_bank = self.labels_bank[:self.index]
        rank1, _, _, mAP = test_map(self.text_bank, self.labels_bank, self.images_bank[::2], self.labels_bank[::2])
        self.log("val_rank1", rank1)

    def forward(self, img, txt, mask):
        """
        img: 4D tensor (B,C=3,H=384,W=128) torch.FloatTensor [0.0, 1.0]
        txt: 2D tensor (B,L) longtensor
        mask: (attention mask) 2D tensor (B,L) longtensor
        """
        img_f4, txt_f4 = self._forward(img, txt, mask)
        return (img_f4, txt_f4)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.adam_lr, weight_decay=self.args.wd)
        scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, int(self.args.epoches_decay), gamma=self.args.lr_decay_ratio)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.args.warm_epoch, after_scheduler=scheduler_steplr)
        # return optimizer
        return [optimizer], [scheduler_warmup]

