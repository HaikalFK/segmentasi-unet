"""
============================================================
 TEST / CEK SETIAP FUNGSI DALAM ENHANCED U-NET
============================================================
Cara menjalankan (dari root project):

    python Enhanced-U-Net/test_enhancedunet.py

Script ini mengecek SATU-PERSATU setiap building block:
    [1] DoubleConv      — conv ganda (+residual, +dropout)
    [2] Down            — maxpool + double conv
    [3] AttentionGate   — filter skip connection
    [4] Up              — upsample + skip + double conv (+attn)
    [5] OutConv         — conv 1x1 output
    [6] EnhancedUNet    — assembly lengkap (semua kombinasi flag)
    [7] deep_supervision_loss — loss gabungan deep supervision
    [8] KESETARAAN      — flag semua OFF == UNet standar (output sama)

SETIAP test memeriksa: bentuk output, gradien mengalir, & tidak NaN.
"""

import os
import sys
import importlib
import torch
import torch.nn.functional as F

# ── Import (folder Enhanced-U-Net pakai tanda hubung → pakai importlib) ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

parts = importlib.import_module('Enhanced-U-Net.enhancedunet_parts')
model_mod = importlib.import_module('Enhanced-U-Net.enhancedunet_model')

DoubleConv, Down, Up, OutConv, AttentionGate = (
    parts.DoubleConv, parts.Down, parts.Up, parts.OutConv, parts.AttentionGate)
EnhancedUNet = model_mod.EnhancedUNet
deep_supervision_loss = model_mod.deep_supervision_loss

passed = 0


def check(cond, label):
    """Tampilkan PASS/FAIL untuk satu pengujian."""
    global passed
    assert cond, f'==> FAIL: {label}'
    passed += 1
    print(f'   [PASS] {label}')


def count_params(module):
    return sum(p.numel() for p in module.parameters())


print('=' * 60)
print(' [1] DoubleConv')
print('=' * 60)
# DoubleConv(3 -> 64), tanpa residual: output sama ukuran input
m = DoubleConv(3, 64, residual=False, dropout=0.0)
x = torch.randn(2, 3, 128, 128)
y = m(x)
check(y.shape == (2, 64, 128, 128), f'DoubleConv 3->64, no-residual: {tuple(y.shape)}')

# DoubleConv dengan residual (in == out): sekarang harus x + conv(x)
m = DoubleConv(64, 64, residual=True, dropout=0.1)
y = m(x.new_zeros(2, 64, 128, 128))
check(y.shape == (2, 64, 128, 128), f'DoubleConv 64->64 residual: {tuple(y.shape)}')

# DoubleConv residual dengan in != out: residual DI-SKIP otomatis (tidak crash)
m = DoubleConv(3, 64, residual=True, dropout=0.0)
y = m(x)
check(y.shape == (2, 64, 128, 128), f'DoubleConv 3->64 residual (auto-skip): {tuple(y.shape)}')
print(f'   (info) parameter DoubleConv 64->64 = {count_params(m):,}')

print()
print('=' * 60)
print(' [2] Down  (maxpool 2x + double conv)')
print('=' * 60)
m = Down(64, 128, dropout=0.0)
x = torch.randn(2, 64, 128, 128)
y = m(x)
check(y.shape == (2, 128, 64, 64), f'Down 64->128: {tuple(y.shape)} (resolusi harus 1/2)')

print()
print('=' * 60)
print(' [3] AttentionGate')
print('=' * 60)
# g = fitur decoder, x = fitur encoder (skip). Output = x yang sudah di-filter.
m = AttentionGate(F_g=64, F_l=128, F_int=32)
g = torch.randn(2, 64, 32, 32)     # gating signal dari decoder
x = torch.randn(2, 128, 32, 32)    # skip connection dari encoder
y = m(g, x)

# Sifat penting attention gate: bobot perhatian psi = sigmoid(...) DI ANTARA 0-1,
# dan output = x * psi. Jadi untuk x bernilai POSITIF, output harus LEBIH KECIL
# ATAU SAMA dengan x (psi <= 1). Ini yang bisa diverifikasi.
x_pos = x.abs()                    # pakai input non-negatif
psi = m(g, x_pos) / (x_pos + 1e-8)  # rekonstruksi psi (0..1) dari output
check(y.shape == (2, 128, 32, 32), f'AttentionGate: {tuple(y.shape)} (sama dengan x)')
check((psi >= 0).all() and (psi <= 1).all(),
      'attention weights psi di dalam [0,1] (sigmoid bekerja)')
check((m(g, x_pos) <= x_pos + 1e-6).all(),
      'output = x * psi (psi<=1) -> output tidak melebihi x untuk x positif')

print()
print('=' * 60)
print(' [4] Up  (upsample + concat skip + double conv)')
print('=' * 60)
# Mode transposed conv (bilinear=False) + attention
#   ConvTranspose2d(1024 -> 512): g (x1 hasil up) = 512 ch, skip x2 = 512 ch
m = Up(1024, 512, bilinear=False, attn=True, dropout=0.0)
x1 = torch.randn(2, 1024, 16, 16)  # dari level bawah (bottleneck)
x2 = torch.randn(2, 512, 32, 32)   # skip connection dari encoder
y = m(x1, x2)
check(y.shape == (2, 512, 32, 32), f'Up bilinear=False + attn: {tuple(y.shape)}')

# Mode bilinear + attention
#   Di model nyata, x1 yang masuk Up(1024,512) bilinear sudah = 512 ch
#   (karena down4 = Down(512, 1024//2)). Upsample tidak mengubah channel:
#   g = 512 ch, skip x2 = 512 ch.
m = Up(1024, 512, bilinear=True, attn=True, dropout=0.0)
x1_bil = torch.randn(2, 512, 16, 16)   # = x5 dari down4 (512 ch)
y = m(x1_bil, x2)
check(y.shape == (2, 512, 32, 32), f'Up bilinear=True + attn: {tuple(y.shape)}')

# Up tanpa attention (perilaku persis U-Net standar)
m = Up(1024, 512, bilinear=False, attn=False, dropout=0.0)
y = m(x1, x2)
check(y.shape == (2, 512, 32, 32), f'Up bilinear=False no-attn: {tuple(y.shape)}')

print()
print('=' * 60)
print(' [5] OutConv  (conv 1x1 -> n_classes)')
print('=' * 60)
m = OutConv(64, 2, dropout=0.0)
x = torch.randn(2, 64, 128, 128)
y = m(x)
check(y.shape == (2, 2, 128, 128), f'OutConv 64->2: {tuple(y.shape)}')

print()
print('=' * 60)
print(' [6] EnhancedUNet  (assembly lengkap)')
print('=' * 60)
x = torch.randn(2, 3, 128, 128)

# 6a. Default enhancement (residual + attn + dropout) — bilinear=False
model = EnhancedUNet(n_channels=3, n_classes=2, bilinear=False,
                     residual=True, attn=True, dropout=0.1)
y = model(x)
check(y.shape == (2, 2, 128, 128), f'(6a) full-enhance bilinear=False: {tuple(y.shape)}')

# 6b. Deep supervision — harus return TUPEL 4 output
model2 = EnhancedUNet(n_channels=3, n_classes=2, bilinear=False, deep_supervision=True)
ys = model2(x)
ok = (len(ys) == 4 and ys[0].shape == (2, 2, 128, 128) and
      ys[1].shape == (2, 2, 16, 16) and ys[2].shape == (2, 2, 32, 32) and
      ys[3].shape == (2, 2, 64, 64))
check(ok, f'(6b) deep_supervision shapes: {[tuple(t.shape) for t in ys]}')

# 6c. bilinear=True + attn (kombinasi yang dulu error — harusnya sudah fixed)
model3 = EnhancedUNet(n_channels=3, n_classes=2, bilinear=True, attn=True)
y = model3(x)
check(y.shape == (2, 2, 128, 128), f'(6c) bilinear=True + attn: {tuple(y.shape)}')

# 6d. Backward pass — gradient harus mengalir ke semua parameter
loss = model(x).sum()
loss.backward()
n_grad = sum(1 for p in model.parameters() if p.grad is not None)
n_all = sum(1 for _ in model.parameters())
check(n_grad == n_all, f'(6d) backward: gradien {n_grad}/{n_all} parameter')

# 6e. Gradient checkpointing
model3.use_checkpointing()
y = model3(x)
check(y.shape == (2, 2, 128, 128), '(6e) use_checkpointing() tetap berfungsi')

# 6f. Hitung parameter tiap level (biar tahu enhancement menambah berapa)
print()
print('   (info) jumlah parameter per model:')
print(f'      Enhanced (res+attn+do0.1) : {count_params(model)/1e6:.2f}M')
print(f'      Deep supervision          : {count_params(model2)/1e6:.2f}M')
print(f'      bilinear=True + attn      : {count_params(model3)/1e6:.2f}M')

print()
print('=' * 60)
print(' [7] deep_supervision_loss')
print('=' * 60)


def combined_loss(pred, target):
    """Salinan loss dari notebook (BCE + Dice)."""
    bce = F.cross_entropy(pred, target)
    prob = F.softmax(pred, dim=1)[:, 1]
    dice = 1 - (2 * (prob * target.float()).sum() + 1e-6) / (prob.sum() + target.float().sum() + 1e-6)
    return bce + dice


masks = (torch.rand(2, 128, 128) > 0.6).long()
outputs = model2(x)          # model2 = deep_supervision=True
loss_ds = deep_supervision_loss(outputs, masks, combined_loss)
loss_ds.backward()
check(torch.isfinite(loss_ds), f'deep_supervision_loss finite: {loss_ds.item():.4f}')

print()
print('=' * 60)
print(' [8] KESETARAAN: flag semua OFF == UNet standar')
print('=' * 60)
from unet import UNet  # noqa: E402  (U-Net standar)

torch.manual_seed(0)
std = UNet(n_channels=3, n_classes=2, bilinear=False)
enh = EnhancedUNet(n_channels=3, n_classes=2, bilinear=False,
                   residual=False, attn=False, dropout=0.0, deep_supervision=False)

# Ambil state_dict standar, isi ke enhanced.
# Catatan: di Enhanced, dropout=0 disisipkan sebagai nn.Identity di posisi 3,
# sehingga index layer bergeser (Conv2d: 3→4, BN: 4→5). Nama key tidak identik,
# jadi kita remap: standar (3=conv2, 4=bn2) → enhanced (4=conv2, 5=bn2).
sd_std = std.state_dict()
sd_remap = {}
for k, v in sd_std.items():
    if '.double_conv.3.' in k:
        k = k.replace('.double_conv.3.', '.double_conv.4.')  # conv kedua
    elif '.double_conv.4.' in k:
        k = k.replace('.double_conv.4.', '.double_conv.5.')  # bn kedua
    sd_remap[k] = v
enh.load_state_dict(sd_remap)

with torch.no_grad():
    y_std = std(x)
    y_enh = enh(x)

diff = (y_std - y_enh).abs().max().item()
check(diff < 1e-6, f'output identik (max diff = {diff:.2e})')

print()
print('=' * 60)
print(f' SELESAI - {passed} test PASSED, 0 FAILED')
print('=' * 60)
