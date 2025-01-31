import torch
import torch.nn.functional as F

# ####
# def xentropy_loss(true, pred, reduction="mean"):
#     """Cross entropy loss. Assumes NHWC!
#
#     Args:
#         pred: prediction array
#         true: ground truth array
#
#     Returns:
#         cross entropy loss
#
#     """
#     epsilon = 10e-8
#     # scale preds so that the class probs of each sample sum to 1
#     pred = pred / torch.sum(pred, -1, keepdim=True)
#     # manual computation of crossentropy
#     pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
#     loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
#     loss = loss.mean() if reduction == "mean" else loss.sum()
#     return loss


####
def xentropy_loss(true, pred, focus=None):
    """Cross entropy loss. Assumes NHWC!

    Args:
        pred: prediction array
        true: ground truth array

    Returns:
        cross entropy loss

    """
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, -1, keepdim=True)
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)

    # cal loss with foucs region
    if isinstance(focus, torch.Tensor):
        # focus = (focus[..., None]).float()
        weight_map = torch.zeros_like(focus)
        fore = torch.sum(focus != 0)
        back = torch.sum(focus == 0)
        weight_map[focus == 0] = fore / (fore + back)
        weight_map[focus != 0] = back / (fore + back)
        total = torch.sum(weight_map)
        weight_map = torch.div(weight_map, total)
        loss = loss * weight_map.unsqueeze(-1)
        loss = loss.sum()
    else:
        loss = loss.mean()
    return loss

####
def dice_loss(true, pred, smooth=1e-3):
    """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
    inse = torch.sum(pred * true, (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss


####
def mse_loss(true, pred, focus=None):
    """Calculate mean squared error loss.

    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps
        foucs: cal loss on foucs area

    Returns:
        loss: mean squared error

    """
    loss = pred - true
    if isinstance(focus, torch.Tensor):
        # focus = (focus[..., None]).float()
        loss = (loss * loss * focus)
        loss = loss.sum() / (focus.sum() + 10e-8)
    else:
        loss = (loss * loss).mean()
    return loss

# ####
# def mse_loss(true, pred):
#     """Calculate mean squared error loss.
#
#     Args:
#         true: ground truth of combined horizontal
#               and vertical maps
#         pred: prediction of combined horizontal
#               and vertical maps
#
#     Returns:
#         loss: mean squared error
#
#     """
#     loss = pred - true
#     loss = (loss * loss).mean()
#     return loss


####
def msge_loss(true, pred, focus, device='cuda:1'):
    """Calculate the mean squared error of the gradients of
    horizontal and vertical map predictions. Assumes
    channel 0 is Vertical and channel 1 is Horizontal.

    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)

    Returns:
        loss:  mean squared error of gradients

    """

    def get_sobel_kernel(size):
        """Get sobel kernel with a given size."""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
            )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
            )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    ####
    def get_gradient_hv(hv):
        """For calculating gradient."""
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    focus = (focus[..., None]).float()  # assume input NHW
    focus = torch.cat([focus, focus], axis=-1)
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focused region
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss

def orthogonal_loss(memory_embedding):

    memory_embedding = F.normalize(memory_embedding, 2, -1)
    H = memory_embedding.shape[0]
    memory_embedding = memory_embedding.unsqueeze(0)
    weight_squared = torch.bmm(memory_embedding, memory_embedding.permute(0, 2, 1))  # (N * C) * H * H
    ones = torch.ones(1, H, H, dtype=torch.float32).cuda()  # 1 * H * H
    diag = torch.eye(H, dtype=torch.float32).cuda()  # 1 * H * H
    loss_orth = ((weight_squared * (ones - diag)) ** 2).sum()
    # average
    loss_orth /= (H * H - H)

    return loss_orth