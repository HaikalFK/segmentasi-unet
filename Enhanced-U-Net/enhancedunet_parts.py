""" Parts of the Enhanced U-Net model

Versi ini adalah "U-Net standar" (lihat unet/unet_parts.py) yang ditambah
enhancement — struktur & tipe kode sengaja dibuat MIRIP persis U-Net standar
agar mudah dibandingkan:

    DoubleConv  + residual skip (ResNet-style, opsional)
    Down        + dropout (opsional)
    Up          + attention gate (opsional)
    AttentionGate (baru — hanya dipakai jika attn=True)
    OutConv     + dropout (opsional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2  (+ optional residual skip connection)

    Sama seperti DoubleConv U-Net standar, hanya:
      - mid_channels default = out_channels (identik)
      - residual=True  → keluar dengan menambah input (x + conv(x))
      - dropout > 0    → dropout di tengah (antara conv pertama & kedua)
    """

    def __init__(self, in_channels, out_channels, mid_channels=None,
                 residual=False, dropout=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.residual = residual
        self.do_residual = residual and (in_channels == out_channels)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.do_residual:
            return self.double_conv(x) + x
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv (+ optional dropout)"""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttentionGate(nn.Module):
    """Attention gate (Oktay et al., 2018) — dipasang di skip connection.

    Mem-filter fitur encoder (x) berdasarkan fitur decoder (g) sebelum
    di-concat: bagian yang "penting" diperkuat, yang tidak relevan diperlemah.

    x  : fitur dari encoder (skip connection), shape (B, F_l, H, W)
    g  : fitur dari decoder (gating signal), shape (B, F_g, H, W)
    out: x yang sudah dip-filter, shape sama dengan x
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g & x harus sama resolusi → kita samakan x ke g
        diffY = g.size()[2] - x.size()[2]
        diffX = g.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class Up(nn.Module):
    """Upscaling then double conv (+ optional attention gate)

    Struktur & argumen identik dengan Up standar, tambahan:
      - attn=True   → pasang AttentionGate di skip connection (x2) sebelum concat
      - dropout>0   → dropout di dalam DoubleConv decoder
    """

    def __init__(self, in_channels, out_channels, bilinear=True,
                 attn=False, dropout=0.0):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # mid_channels = in_channels//2 (sama seperti U-Net standar),
            # dipanggil dengan KEYWORD AGAR tidak bentrok dengan argumen residual.
            self.conv = DoubleConv(in_channels, out_channels,
                                   mid_channels=in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

        self.attn = attn
        if attn:
            # Gating signal (g = x1 HASIL up) dan skip (x = x2) KEDUANYA punya
            # in_channels // 2 channel, di KEDUA mode upsampling:
            #   - bilinear=True : down4 = Down(512, 1024//2) → x5 = 512 ch,
            #                     Upsample tidak mengubah channel → g = 512 ch
            #   - bilinear=False: ConvTranspose2d(1024 → 512) → g = 512 ch
            # sedangkan skip (x2 = encoder level) juga selalu in//2 channel.
            # Jadi F_g = F_l = in_channels // 2, konsisten di semua level.
            F_g = in_channels // 2
            F_l = in_channels // 2
            self.att_gate = AttentionGate(F_g=F_g, F_l=F_l, F_int=F_l // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        # attention gate dipasang DI SINI — di skip connection, sebelum concat
        if self.attn:
            x2 = self.att_gate(x1, x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.conv(x))
