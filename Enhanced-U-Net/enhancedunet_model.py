""" Full assembly of the parts to form the complete Enhanced U-Net network

Struktur & argumen konstruktor MIRIP persis UNet standar (unet/unet_model.py):
    UNet(n_channels, n_classes, bilinear)
Hanya dibungkus oleh flag enhancement (semua opsional, default OFF = perilaku
persis U-Net standar):
    - residual  : DoubleConv pakai residual skip (ResNet-style)
    - attn      : attention gate di semua skip connection
    - dropout   : dropout rate untuk regularisasi (0 = mati)
    - deep_supervision : output auxiliary loss di tiap level decoder
"""

from .enhancedunet_parts import *


class EnhancedUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,
                 residual=True, attn=True, dropout=0.0, deep_supervision=False):
        super(EnhancedUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision
        self._checkpointing = False

        self.inc = (DoubleConv(n_channels, 64, residual=residual, dropout=dropout))
        self.down1 = (Down(64, 128, dropout=dropout))
        self.down2 = (Down(128, 256, dropout=dropout))
        self.down3 = (Down(256, 512, dropout=dropout))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, dropout=dropout))
        self.up1 = (Up(1024, 512 // factor, bilinear, attn, dropout))
        self.up2 = (Up(512, 256 // factor, bilinear, attn, dropout))
        self.up3 = (Up(256, 128 // factor, bilinear, attn, dropout))
        self.up4 = (Up(128, 64, bilinear, attn, dropout))
        self.outc = (OutConv(64, n_classes, dropout=dropout))

        # Deep supervision — auxiliary output per level decoder (opsional)
        if deep_supervision:
            self.ds1 = OutConv(512 // factor, n_classes)   # up1 → resolusi 1/8
            self.ds2 = OutConv(256 // factor, n_classes)   # up2 → resolusi 1/4
            self.ds3 = OutConv(128 // factor, n_classes)   # up3 → resolusi 1/2
            # up4 sudah di-handle outc

    def forward(self, x):
        # Gradient checkpointing — kalau diaktifkan, forward tiap blok dihitung
        # ulang saat backward (hemat memori, tambah komputasi).
        if self._checkpointing:
            cp = lambda fn, *a: torch.utils.checkpoint.checkpoint(fn, *a, use_reentrant=False)
        else:
            cp = lambda fn, *a: fn(*a)

        # ── Encoder path (kontraksi) ──
        x1 = cp(self.inc, x)      # (64,   H,     W)
        x2 = cp(self.down1, x1)   # (128,  H/2,   W/2)
        x3 = cp(self.down2, x2)   # (256,  H/4,   W/4)
        x4 = cp(self.down3, x3)   # (512,  H/8,   W/8)
        x5 = cp(self.down4, x4)   # (1024, H/16,  W/16)

        # ── Decoder path (ekspansi) ──
        d1 = cp(self.up1, x5, x4) # (512/f, H/8,  W/8)
        d2 = cp(self.up2, d1, x3) # (256/f, H/4,  W/4)
        d3 = cp(self.up3, d2, x2) # (128/f, H/2,  W/2)
        d4 = cp(self.up4, d3, x1) # (64,    H,    W)
        logits = self.outc(d4)  # (n_classes, H, W)

        # Deep supervision — return semua auxiliary logits
        if self.deep_supervision:
            return logits, self.ds1(d1), self.ds2(d2), self.ds3(d3)
        return logits

    def use_checkpointing(self):
        """Aktifkan gradient checkpointing.

        CATATAN: versi standar (unet/unet_model.py) menulis
            self.inc = torch.utils.checkpoint(self.inc)
        yang sebenarnya TIDAK berfungsi — torch.utils.checkpoint adalah module,
        bukan fungsi callable (bug yang diwarisi dari repo milesial asli).
        Di sini diperbaiki: flag _checkpointing dipakai di forward di atas.
        """
        self._checkpointing = True
        # use_reentrant=False direkomendasikan & lebih hemat memori (varian non-reentrant)


def deep_supervision_loss(outputs, target, loss_fn, weights=(1.0, 0.5, 0.25, 0.125)):
    """Hitung loss untuk deep supervision.

    outputs : tuple dari EnhancedUNet.forward saat deep_supervision=True
              (logits, aux1, aux2, aux3) dengan resolusi H, H/2, H/4, H/8.
    target  : mask ground-truth berukuran (B, H, W) di resolusi penuh.
    loss_fn : callable(logits, target_upsampled) -> scalar loss.
    weights : bobot tiap output, dari yang paling halus ke paling kasar.

    Contoh penggunaan:
        if model.deep_supervision:
            outputs = model(imgs)
            loss = deep_supervision_loss(outputs, masks, combined_loss)
        else:
            loss = combined_loss(model(imgs), masks)
    """
    total = 0.0
    for out, w in zip(outputs, weights):
        if out.shape[-2:] != target.shape[-2:]:
            out = F.interpolate(out, size=target.shape[-2:], mode='bilinear', align_corners=True)
        total += w * loss_fn(out, target)
    return total
