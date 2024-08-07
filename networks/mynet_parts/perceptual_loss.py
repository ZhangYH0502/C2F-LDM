"""
Source
    https://github.com/IgorSusmelj/pytorch-styleguide/blob/master/building_blocks.md
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
Usage
    Note: This loss has parameters and should therefore be put on GPU if you have a model on GPU.
    We can use the module as a new criterion:

    # define the criterion
    criterion_VGG = VGGLoss()

    # put it on GPU if you can
    criterion_VGG = criterion_VGG.cuda()

    # calc perceptual loss during train loop
    # to compute the perceptual loss of an auto-encoder
    # fake_ae is the output of your auto-encoder
    # img is the original input image
    ae_loss_VGG = criterion_VGG(fake_ae, img)

    # do backward or sum up with other losses...
    ae_loss_VGG.backward()
"""
import torch
import torchvision
class VGGLoss(torch.nn.Module):
  def __init__(self):
    super(VGGLoss, self).__init__()
    self.vgg = Vgg19().cuda()
    self.criterion = torch.nn.L1Loss()
    self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

  def forward(self, x, y):
    x_vgg, y_vgg = self.vgg(x), self.vgg(y)
    loss = 0
    for i in range(len(x_vgg)):
      loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
    return loss


class Vgg19(torch.nn.Module):
  def __init__(self, requires_grad=False):
    super(Vgg19, self).__init__()
    vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
    self.slice1 = torch.nn.Sequential()
    self.slice2 = torch.nn.Sequential()
    self.slice3 = torch.nn.Sequential()
    self.slice4 = torch.nn.Sequential()
    self.slice5 = torch.nn.Sequential()
    for x in range(2):
      self.slice1.add_module(str(x), vgg_pretrained_features[x])
    for x in range(2, 7):
      self.slice2.add_module(str(x), vgg_pretrained_features[x])
    for x in range(7, 12):
      self.slice3.add_module(str(x), vgg_pretrained_features[x])
    for x in range(12, 21):
      self.slice4.add_module(str(x), vgg_pretrained_features[x])
    for x in range(21, 30):
      self.slice5.add_module(str(x), vgg_pretrained_features[x])
    if not requires_grad:
      for param in self.parameters():
        param.requires_grad = False

  def forward(self, X):
    h_relu1 = self.slice1(X)
    h_relu2 = self.slice2(h_relu1)
    h_relu3 = self.slice3(h_relu2)
    h_relu4 = self.slice4(h_relu3)
    h_relu5 = self.slice5(h_relu4)
    out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
    return out