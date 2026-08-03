Membandingkan Loss Function: Menjalankan U-Net dengan Standard BCE vs Weighted BCE + Dice untuk membuktikan secara kuantitatif bahwa fungsi loss kombinasi memang lebih unggul.

rancangannya adalah dengan membuat satu file utlis/loss_functions.py yang berisikan semua lost yang akan dicari agar code lebih modular.

Fase 2: Implementasi 2 Loss Function

┌─────────────────────┬───────────────────────────────┬─────────────────────────────────────────────┐
│        Loss         │           Deskripsi           │               Parameter Kunci               │
├─────────────────────┼───────────────────────────────┼─────────────────────────────────────────────┤
│ Standard BCE        │ F.cross_entropy(pred, target) │ -                                           │
├─────────────────────┼───────────────────────────────┼─────────────────────────────────────────────┤
│ Weighted BCE + Dice │ weighted_ce + dice_loss       │ class_weights (inverse freq), clamp 0.3-3.0 │
└─────────────────────┴───────────────────────────────┴─────────────────────────────────────────────┘