import torch
import torch.nn as nn
import torch.nn.functional as F




def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


#######################


def calc_emd_loss(pred, target):
    b, _, h, w = pred.shape
    pred = pred.view([b, -1, w * h])
    pred_norm = torch.sqrt((pred**2).sum(1).view([b, -1, 1]))
    pred = pred.permute([0, 2, 1])
    target_t = target.view([b, -1, w * h])
    target_norm = torch.sqrt((target**2).sum(1).view([b, 1, -1]))
    similarity = torch.bmm(pred, target_t) / pred_norm / target_norm
    dist = 1. - similarity

    # return's shape is b*(h*w)*(h*w)
    return dist


# calc_style_emd_loss
class CalcStyleEmdLoss():
    def __init__(self):
        super(CalcStyleEmdLoss, self).__init__()

    def __call__(self, pred, target):
        CX_M = calc_emd_loss(pred, target)
        m1 = CX_M.min(2).values.mean()
        m2 = CX_M.min(1).values.mean()
        #m = torch.stack([m1, m2])
        loss_remd = torch.max(m1, m2)

        return loss_remd


# calc_content_relt_loss
class CalcContentReltLoss():
    """Calc Content Relt Loss.
    """
    def __init__(self):
        super(CalcContentReltLoss, self).__init__()

    def __call__(self, pred, target):

        Mx = calc_emd_loss(pred, pred)
        My = calc_emd_loss(target, target)
        loss_content = torch.abs(Mx - My).mean()
        return loss_content



def calc_mean_std(feat, eps=1e-5):
    """calculate mean and standard deviation.
    Args:
        feat (Tensor): Tensor with shape (N, C, H, W).
        eps (float): Default: 1e-5.
    Return:
        mean and std of feat
        shape: [N, C, 1, 1]
    """
    size = feat.shape
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view([N, C, -1])
    feat_var = torch.var(feat_var, dim=2) + eps
    feat_std = torch.sqrt(feat_var)
    feat_std = feat_std.view([N, C, 1, 1])
    feat_mean = feat.view([N, C, -1])
    feat_mean = torch.mean(feat_mean, dim=2)
    feat_mean = feat_mean.reshape([N, C, 1, 1])
    return feat_mean, feat_std


def mean_variance_norm(feat):
    """mean_variance_norm.
    Args:
        feat (Tensor): Tensor with shape (N, C, H, W).
    Return:
        Normalized feat with shape (N, C, H, W)
    """
    size = feat.shape
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


# calc_style_loss
class CalcStyleLoss():
    """Calc Style Loss.
    """
    def __init__(self, mean_std=True):
        self.mse_loss = nn.MSELoss()
        self.mean_std = mean_std

    def __call__(self, pred, target):
        """Forward Function.
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        if self.mean_std:
            pred_mean, pred_std = calc_mean_std(pred)
            target_mean, target_std = calc_mean_std(target)
            return self.mse_loss(pred_mean, target_mean) + self.mse_loss(pred_std, target_std)
        else:
            return self.mse_loss(pred, target)


# calc_content_relt_loss
class CalcContentReltLoss():
    """Calc Content Relt Loss.
    """
    def __init__(self):
        super(CalcContentReltLoss, self).__init__()

    def __call__(self, pred, target):

        Mx = calc_emd_loss(pred, pred)
        My = calc_emd_loss(target, target)
        loss_content = torch.abs(Mx - My).mean()
        return loss_content


# CalcContentLoss
class CalcContentLoss():
    """Calc Content Loss.
    """
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __call__(self, pred, target, norm=False):
        """Forward Function.
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            norm(Bool): whether use mean_variance_norm for pred and target
        """
        if (norm == False):
            return self.mse_loss(pred, target)
        else:
            return self.mse_loss(mean_variance_norm(pred),
                                 mean_variance_norm(target))


def calc_tv_loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """
    def __init__(self,
                 gan_mode,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 loss_weight=1.0):
        """ Initialize the GANLoss class.
        Args:
            gan_mode (str): the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool): label for a real image
            target_fake_label (bool): label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        # when loss weight less than zero return None
        if loss_weight <= 0:
            return None

        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.loss_weight = loss_weight

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgan', 'wgangp', 'hinge', 'logistic']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Args:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
        if target_is_real:
            if not hasattr(self, 'target_real_tensor'):
                self.target_real_tensor = torch.full(
                    prediction.size(),
                    fill_value=self.target_real_label,
                    dtype=torch.float32).cuda()
            target_tensor = self.target_real_tensor
        else:
            if not hasattr(self, 'target_fake_tensor'):
                self.target_fake_tensor = torch.full(
                    prediction.size(),
                    fill_value=self.target_fake_label,
                    dtype=torch.float32).cuda()
            target_tensor = self.target_fake_tensor

        return target_tensor

    def __call__(self,
                 prediction,
                 target_is_real,
                 is_disc=False,
                 is_updating_D=None):
        """Calculate loss given Discriminator's output and grount truth labels.
        Args:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
            is_updating_D (bool)  - - if we are in updating D step or not
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode.find('wgan') != -1:
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if target_is_real:
                loss = F.relu(1 - prediction) if is_updating_D else -prediction
            else:
                loss = F.relu(1 + prediction) if is_updating_D else prediction
            loss = loss.mean()
        elif self.gan_mode == 'logistic':
            if target_is_real:
                loss = F.softplus(-prediction).mean()
            else:
                loss = F.softplus(prediction).mean()

        return loss if is_disc else loss * self.loss_weight