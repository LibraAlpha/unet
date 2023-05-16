import torch
import torch.nn as nn
import torch.nn.functional as F
from model.components import DoubleConv, InConv, Down, Up, OutConv


class Unet(nn.Module):

    def __init__(self, in_ch, out_ch, gpu_ids=None, bilinear=False):  # inch, 图片的通道数，1表示灰度图像，3表示彩色图像
        super(Unet, self).__init__()
        if gpu_ids is None:
            gpu_ids = []
        self.loss = None
        self.matrix_iou = None
        self.pred_y = None
        self.x = None
        self.y = None

        self.loss_stack = 0
        self.matrix_iou_stack = 0
        self.stack_count = 0
        self.display_names = ['loss_stack', 'matrix_iou_stack']

        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if torch.cuda.is_available() else torch.device(
            'cpu')

        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.bce_loss = nn.BCELoss()

        self.inc = (DoubleConv(in_ch, 64))
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)

        self.down3 = Down(256, 512)
        self.drop3 = nn.Dropout2d(0.5)

        self.down4 = Down(512, 1024)
        self.drop4 = nn.Dropout2d(0.5)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)

        self.out = OutConv(64, out_ch)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self):
        x1 = self.inc(self.x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.drop3(x4)
        x5 = self.down4(x4)
        x5 = self.drop4(x5)

        # skip connection与采样结果融合
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        self.pred_y = nn.functional.sigmoid(x)

    def set_input(self, x, y):
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.to(self.device)

    def optimize_params(self):
        self.forward()
        self._bce_iou_loss()
        _ = self.accu_iou()
        self.stack_count += 1
        self.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def accu_iou(self):
        y_pred = (self.pred_y > 0.5) * 1.0
        y_true = (self.y > 0.5) * 1.0

        pred_flat = y_pred.view(y_pred.numel())
        true_flat = y_true.view(y_true.numel())

        intersection = float(torch.sum(pred_flat * true_flat)) + 1e-7
        denominator = float(torch.sum(pred_flat + true_flat)) - intersection + 2e-7

        self.matrix_iou = intersection / denominator
        self.matrix_iou_stack += self.matrix_iou
        return self.matrix_iou

    def _bce_iou_loss(self):
        y_pred = self.pred_y
        y_true = self.y
        pred_flat = y_pred.view(y_pred.numel())
        true_flat = y_true.view(y_true.numel())

        intersection = torch.sum(pred_flat * true_flat) + 1e-7
        denominator = torch.sum(pred_flat + true_flat) - intersection + 1e-7
        iou = torch.div(intersection, denominator)
        bce_loss = self.bce_loss(pred_flat, true_flat)
        self.loss = bce_loss - iou + 1
        self.loss_stack += self.loss

    def get_current_losses(self):
        errors_ret = {}
        for name in self.display_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name)) / self.stack_count
        self.loss_stack = 0
        self.matrix_iou_stack = 0
        self.stack_count = 0
        return errors_ret

    def eval_iou(self):
        with torch.no_grad():
            self.forward()
            self._bce_iou_loss()
            _ = self.accu_iou()
            self.stack_count += 1


if __name__ == '__main__':
    in_ch = 3
    out_ch = 1
    unet = Unet(in_ch, out_ch, gpu_ids=[0])

    x = torch.randn(1, in_ch, 256, 256)
    y = torch.randn(1, out_ch, 256, 256)

    unet.set_input(x, y)

    unet.optimize_params()

    losses = unet.get_current_losses()
    print(losses)