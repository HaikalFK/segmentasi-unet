1. Prediksi hanya menangkap tepi/kontur, bukan area penuh
Prediksi Anda terlihat seperti "coretan" tipis dan berpola speckle (noise) di sekitar tepi daun, bukan blob solid seperti ground truth. Ini pola klasik dari:

Class imbalance parah — piksel background jauh lebih banyak daripada foreground (daun). Kalau memakai Binary Cross-Entropy polos tanpa pembobotan, model cenderung "malas" memprediksi foreground secara solid, dan sinyal gradien terkuat justru ada di area tepi (kontras tinggi), sehingga model hanya "menempel" di situ.
mIoU = 0.0005 ini sangat ekstrem — praktis mendekati nol, hampir setara prediksi acak. Ini bukan sekadar "kurang optimal", ini indikasi model belum benar-benar belajar atau ada bug di pipeline, bukan cuma masalah arsitektur yang kurang bagus.

2. Kemungkinan penyebab teknis (urut dari yang paling mungkin):

Loss tidak turun signifikan saat training (underfitting parah) — cek learning curve Anda
Label/mask tidak sinkron dengan gambar input (misal urutan file, resize mask pakai interpolasi bilinear bukan nearest sehingga label rusak)
Learning rate terlalu tinggi/rendah, atau optimizer belum konvergen
Aktivasi output terakhir salah (misal pakai softmax padahal butuh sigmoid untuk biner, atau threshold 0.5 tidak sesuai skala output)
Ukuran output tidak sama persis dengan ground truth setelah upsampling, sehingga IoU dihitung dari data yang misaligned

3. Catatan soal instance vs semantic segmentation
Ground truth Anda adalah instance segmentation (tiap daun warna berbeda), tapi U-Net standar biasanya hanya bisa keluarkan semantic segmentation (daun vs background sebagai satu kelas). U-Net standar tidak bisa memisahkan antar-instance yang saling menempel — jadi walau nanti hasilnya "solid", ia tidak akan otomatis memisahkan 12 daun ini menjadi 12 label berbeda kecuali Anda pakai pendekatan tambahan.