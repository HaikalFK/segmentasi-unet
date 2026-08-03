Semua file perbandingan sudah tersedia.
Device: cuda
GPU: Tesla T4

============================================================
EXPERIMENT: standard_bce
============================================================
Dataset: 347 samples, binary (bg + leaf)
Train: 313  Val: 34
[wandb] init failed (init() got an unexpected keyword argument 'finish_previous') — continuing without tracking
Epochs: 100, Early stop patience: 15, Delta: 0.005
------------------------------------------------------------
Epoch   1/100 | Loss: 0.1919 | Val Loss: 2.4032 | IoU: 0.0103 | Dice: 0.0199
  ★ New best Dice: 0.0199
Epoch   2/100 | Loss: 0.0802 | Val Loss: 0.4662 | IoU: 0.2570 | Dice: 0.4048
  ★ New best Dice: 0.4048
Epoch   3/100 | Loss: 0.0639 | Val Loss: 1.2329 | IoU: 0.3229 | Dice: 0.4719
  ★ New best Dice: 0.4719
Epoch   4/100 | Loss: 0.0637 | Val Loss: 0.1557 | IoU: 0.8265 | Dice: 0.9047
  ★ New best Dice: 0.9047
Epoch   5/100 | Loss: 0.0581 | Val Loss: 0.2994 | IoU: 0.7054 | Dice: 0.8255
Epoch   6/100 | Loss: 0.0504 | Val Loss: 0.3891 | IoU: 0.6698 | Dice: 0.8007
Epoch   7/100 | Loss: 0.0481 | Val Loss: 0.0874 | IoU: 0.8921 | Dice: 0.9430
  ★ New best Dice: 0.9430
Epoch   8/100 | Loss: 0.0485 | Val Loss: 0.0635 | IoU: 0.8928 | Dice: 0.9433
Epoch   9/100 | Loss: 0.0459 | Val Loss: 0.0302 | IoU: 0.9100 | Dice: 0.9528
  ★ New best Dice: 0.9528
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2013_plant068', 'ara2013_plant028', 'ara2013_plant008', 'tobacco_plant007', 'ara2013_plant153', 'ara2013_plant110', 'ara2013_plant115', 'ara2013_plant159'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['ara2013_plant090', 'ara2013_plant136', 'ara2012_plant009', 'ara2013_plant005', 'ara2013_plant138', 'ara2013_plant134', 'ara2013_plant129', 'ara2013_plant149'] — 1048576 elemen.
Epoch  10/100 | Loss: 0.0448 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (D)] IoU anjlok drastis: 0.9100 → 0.0000 di epoch 10. Cek prediksi di results/standard_bce/predictions.
[WARN (B)] grad NaN at bi=39 names=['tobacco_plant007'] — step di-skip.
[INFO] Epoch 11: 1 batch di-skip karena NaN/Inf (diagnosis B/C).
Epoch  11/100 | Loss: 0.0455 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2013_plant090', 'ara2013_plant124', 'tobacco_plant023', 'ara2012_plant089', 'ara2013_plant034', 'ara2013_plant125', 'ara2013_plant013', 'ara2012_plant030'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['ara2013_plant140', 'ara2013_plant110', 'ara2013_plant149', 'ara2013_plant073', 'ara2013_plant080', 'ara2013_plant151', 'ara2013_plant094', 'ara2013_plant127'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['tobacco_plant021', 'ara2013_plant041', 'ara2013_plant072', 'ara2013_plant144', 'ara2013_plant096', 'ara2013_plant068', 'ara2012_plant114', 'tobacco_plant030'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=20 names=['ara2012_plant024', 'ara2013_plant052', 'ara2013_plant069', 'ara2013_plant112', 'tobacco_plant044', 'tobacco_plant060', 'ara2013_plant008', 'tobacco_plant035'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2013_plant066', 'ara2012_plant117', 'ara2012_plant104', 'ara2013_plant007', 'tobacco_plant058', 'ara2013_plant128', 'ara2013_plant164', 'tobacco_plant039'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['ara2013_plant136', 'ara2013_plant148', 'ara2013_plant030', 'ara2013_plant135', 'ara2012_plant020', 'ara2012_plant023', 'tobacco_plant040', 'ara2012_plant021'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2013_plant010', 'ara2013_plant145', 'ara2013_plant016', 'ara2013_plant116', 'tobacco_plant015', 'ara2012_plant046', 'tobacco_plant062', 'ara2013_plant075'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2012_plant015', 'ara2013_plant009', 'tobacco_plant006', 'ara2013_plant157', 'ara2012_plant005', 'tobacco_plant022', 'ara2012_plant090', 'ara2013_plant103'] — 1048576 elemen.
Epoch  12/100 | Loss: 0.0436 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=0 names=['tobacco_plant007', 'ara2012_plant076', 'ara2013_plant005', 'ara2013_plant045', 'ara2013_plant096', 'ara2012_plant029', 'ara2013_plant011', 'ara2013_plant111'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['tobacco_plant040', 'ara2012_plant017', 'ara2013_plant101', 'ara2013_plant021', 'tobacco_plant021', 'ara2013_plant157', 'ara2012_plant044', 'ara2013_plant120'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2013_plant153', 'ara2013_plant128', 'tobacco_plant023', 'ara2013_plant014', 'tobacco_plant011', 'ara2012_plant108', 'ara2013_plant126', 'ara2012_plant104'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['tobacco_plant006', 'ara2013_plant102', 'tobacco_plant010', 'ara2013_plant125', 'tobacco_plant019', 'ara2013_plant117', 'ara2013_plant073', 'ara2013_plant040'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['ara2013_plant109', 'tobacco_plant033', 'ara2013_plant094', 'ara2013_plant129', 'ara2012_plant035', 'ara2012_plant062', 'ara2012_plant011', 'ara2013_plant044'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2012_plant016', 'tobacco_plant045', 'tobacco_plant036', 'ara2013_plant029', 'ara2013_plant033', 'ara2013_plant140', 'tobacco_plant050', 'ara2012_plant001'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2012_plant019', 'ara2013_plant118', 'ara2013_plant047', 'ara2013_plant158', 'ara2012_plant056', 'ara2013_plant080', 'ara2013_plant039', 'ara2013_plant007'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['ara2012_plant047', 'tobacco_plant014', 'ara2013_plant149', 'ara2013_plant066', 'ara2012_plant109', 'ara2012_plant116', 'tobacco_plant041', 'ara2012_plant028'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2013_plant068', 'ara2013_plant090', 'ara2013_plant031', 'tobacco_plant062', 'ara2013_plant030', 'tobacco_plant028', 'tobacco_plant017', 'ara2013_plant159'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2013_plant121', 'ara2013_plant064', 'ara2013_plant053', 'ara2012_plant008', 'ara2013_plant032', 'ara2013_plant105', 'ara2013_plant070', 'ara2013_plant123'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['ara2012_plant059', 'ara2013_plant076', 'ara2012_plant036', 'ara2012_plant087', 'ara2013_plant019', 'ara2012_plant031', 'ara2012_plant048', 'ara2013_plant160'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2013_plant020', 'tobacco_plant032', 'ara2013_plant137', 'tobacco_plant051', 'ara2012_plant018', 'ara2013_plant010', 'ara2012_plant039', 'ara2013_plant051'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['tobacco_plant042', 'ara2013_plant026', 'ara2013_plant106', 'ara2013_plant082', 'ara2012_plant074', 'ara2013_plant115', 'ara2012_plant046', 'ara2013_plant104'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['ara2012_plant006', 'ara2013_plant027', 'ara2013_plant092', 'ara2012_plant075', 'ara2012_plant003', 'ara2013_plant037', 'tobacco_plant001', 'ara2012_plant045'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2012_plant004', 'ara2013_plant148', 'tobacco_plant037', 'ara2013_plant067', 'tobacco_plant013', 'ara2012_plant071', 'ara2012_plant013', 'tobacco_plant015'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['ara2012_plant089', 'ara2012_plant079', 'tobacco_plant038', 'ara2013_plant161', 'ara2013_plant099', 'ara2013_plant163', 'ara2013_plant057', 'ara2012_plant073'] — 1048576 elemen.
Epoch  13/100 | Loss: 0.0439 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=0 names=['ara2012_plant111', 'tobacco_plant046', 'ara2013_plant053', 'ara2013_plant025', 'ara2012_plant068', 'ara2013_plant099', 'ara2013_plant104', 'ara2013_plant112'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=1 names=['ara2013_plant102', 'ara2012_plant102', 'ara2013_plant082', 'ara2013_plant134', 'ara2013_plant063', 'ara2012_plant036', 'tobacco_plant034', 'ara2013_plant045'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['ara2013_plant163', 'ara2012_plant028', 'ara2012_plant073', 'ara2013_plant020', 'ara2013_plant056', 'ara2012_plant093', 'ara2012_plant054', 'tobacco_plant001'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=8 names=['ara2013_plant145', 'ara2013_plant114', 'tobacco_plant002', 'ara2012_plant094', 'ara2013_plant036', 'tobacco_plant040', 'ara2012_plant030', 'ara2012_plant107'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['tobacco_plant015', 'tobacco_plant036', 'ara2013_plant089', 'ara2013_plant067', 'ara2013_plant061', 'ara2013_plant106', 'ara2013_plant165', 'ara2012_plant008'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2013_plant153', 'ara2012_plant088', 'ara2013_plant042', 'ara2012_plant022', 'ara2013_plant041', 'tobacco_plant030', 'ara2012_plant020', 'ara2013_plant126'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2012_plant037', 'ara2012_plant001', 'tobacco_plant024', 'ara2013_plant096', 'ara2013_plant023', 'ara2012_plant078', 'ara2013_plant160', 'ara2013_plant157'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['tobacco_plant006', 'ara2013_plant062', 'ara2012_plant032', 'tobacco_plant013', 'ara2013_plant092', 'ara2013_plant143', 'ara2012_plant009', 'tobacco_plant060'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2013_plant054', 'ara2013_plant118', 'tobacco_plant021', 'ara2013_plant136', 'ara2012_plant015', 'ara2013_plant085', 'ara2013_plant019', 'tobacco_plant045'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2013_plant125', 'ara2012_plant013', 'ara2012_plant050', 'ara2012_plant016', 'tobacco_plant009', 'ara2012_plant024', 'ara2012_plant099', 'ara2013_plant156'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['ara2013_plant029', 'ara2013_plant076', 'ara2013_plant039', 'ara2013_plant012', 'ara2013_plant117', 'ara2013_plant007', 'ara2013_plant033', 'ara2013_plant155'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['tobacco_plant032', 'ara2013_plant124', 'ara2013_plant080', 'tobacco_plant014', 'ara2013_plant026', 'ara2013_plant022', 'ara2013_plant101', 'ara2013_plant108'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=32 names=['tobacco_plant041', 'tobacco_plant017', 'ara2012_plant044', 'ara2012_plant075', 'ara2012_plant055', 'ara2013_plant127', 'ara2012_plant114', 'ara2012_plant056'] — 1048576 elemen.
Epoch  14/100 | Loss: 0.0433 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=2 names=['ara2013_plant159', 'ara2012_plant062', 'ara2013_plant142', 'ara2013_plant089', 'ara2012_plant043', 'ara2012_plant090', 'ara2013_plant145', 'ara2013_plant014'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2013_plant071', 'ara2013_plant117', 'ara2013_plant105', 'tobacco_plant024', 'tobacco_plant060', 'ara2013_plant055', 'ara2012_plant116', 'ara2013_plant112'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=9 names=['ara2013_plant040', 'tobacco_plant016', 'ara2012_plant029', 'ara2013_plant161', 'ara2013_plant125', 'ara2012_plant053', 'ara2013_plant053', 'ara2013_plant047'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['ara2012_plant109', 'ara2012_plant011', 'ara2012_plant099', 'ara2013_plant032', 'tobacco_plant004', 'tobacco_plant046', 'ara2013_plant049', 'tobacco_plant021'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['ara2012_plant006', 'ara2013_plant010', 'ara2013_plant041', 'ara2012_plant060', 'ara2012_plant120', 'ara2012_plant119', 'ara2013_plant026', 'ara2012_plant111'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['tobacco_plant051', 'ara2013_plant130', 'ara2013_plant120', 'tobacco_plant059', 'ara2013_plant085', 'ara2013_plant157', 'ara2012_plant095', 'ara2013_plant127'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['ara2012_plant021', 'ara2013_plant148', 'ara2013_plant133', 'ara2012_plant068', 'ara2013_plant121', 'ara2012_plant084', 'ara2012_plant059', 'ara2012_plant048'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2012_plant078', 'tobacco_plant018', 'ara2012_plant076', 'tobacco_plant053', 'ara2013_plant149', 'ara2013_plant016', 'ara2013_plant129', 'ara2012_plant037'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['tobacco_plant039', 'ara2013_plant082', 'ara2013_plant165', 'tobacco_plant062', 'tobacco_plant006', 'ara2012_plant013', 'ara2012_plant018', 'ara2013_plant114'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['ara2013_plant136', 'ara2013_plant027', 'ara2013_plant065', 'tobacco_plant014', 'ara2013_plant078', 'tobacco_plant032', 'ara2013_plant038', 'ara2012_plant008'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2012_plant069', 'tobacco_plant001', 'ara2012_plant039', 'ara2013_plant080', 'ara2013_plant158', 'tobacco_plant047', 'ara2013_plant132', 'ara2012_plant097'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['ara2012_plant022', 'ara2013_plant062', 'ara2013_plant056', 'ara2013_plant030', 'tobacco_plant045', 'ara2012_plant042', 'ara2012_plant045', 'ara2013_plant006'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2012_plant019', 'ara2012_plant024', 'ara2012_plant101', 'ara2013_plant094', 'ara2012_plant055', 'ara2013_plant034', 'tobacco_plant040', 'ara2013_plant118'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['ara2013_plant100', 'ara2013_plant144', 'ara2012_plant071', 'ara2013_plant154', 'tobacco_plant030', 'ara2013_plant043', 'ara2012_plant058', 'ara2013_plant046'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2012_plant036', 'ara2013_plant039', 'ara2013_plant035', 'ara2013_plant138', 'ara2012_plant028', 'ara2013_plant115', 'ara2013_plant019', 'ara2012_plant077'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['tobacco_plant013', 'tobacco_plant010', 'ara2012_plant005', 'tobacco_plant038', 'ara2012_plant056', 'ara2012_plant072', 'ara2013_plant021', 'ara2012_plant079'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['tobacco_plant037', 'ara2013_plant068', 'tobacco_plant061', 'ara2013_plant076', 'ara2012_plant113', 'tobacco_plant015', 'ara2013_plant037', 'ara2013_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2013_plant084', 'ara2013_plant156', 'tobacco_plant005', 'tobacco_plant033', 'ara2013_plant044', 'ara2013_plant155', 'ara2013_plant090', 'ara2012_plant033'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['ara2013_plant009', 'ara2012_plant065', 'ara2013_plant022', 'ara2013_plant081', 'ara2012_plant094', 'tobacco_plant012', 'ara2012_plant103', 'ara2012_plant102'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2013_plant067', 'tobacco_plant029', 'tobacco_plant011', 'ara2013_plant064', 'ara2013_plant036', 'ara2013_plant054', 'tobacco_plant025', 'ara2013_plant098'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=36 names=['ara2013_plant073', 'ara2013_plant091', 'ara2012_plant030', 'ara2013_plant086', 'ara2013_plant069', 'ara2013_plant020', 'ara2013_plant072', 'ara2012_plant074'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['ara2013_plant033', 'ara2013_plant164', 'tobacco_plant041', 'ara2013_plant137', 'ara2013_plant104', 'ara2013_plant097', 'ara2013_plant106', 'ara2012_plant107'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2013_plant139', 'ara2012_plant089', 'ara2012_plant020', 'ara2012_plant007', 'ara2013_plant151', 'ara2013_plant123', 'ara2012_plant104', 'ara2013_plant134'] — 1048576 elemen.
Epoch  15/100 | Loss: 0.0447 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=1 names=['tobacco_plant035', 'tobacco_plant037', 'ara2012_plant078', 'ara2012_plant113', 'ara2012_plant102', 'ara2013_plant146', 'ara2012_plant090', 'ara2013_plant034'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['ara2013_plant085', 'ara2013_plant111', 'ara2013_plant044', 'ara2013_plant155', 'ara2013_plant021', 'ara2013_plant143', 'ara2013_plant130', 'ara2013_plant139'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['tobacco_plant004', 'ara2013_plant033', 'ara2013_plant115', 'ara2013_plant036', 'tobacco_plant049', 'tobacco_plant028', 'ara2013_plant165', 'tobacco_plant026'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2013_plant105', 'ara2013_plant067', 'ara2012_plant027', 'tobacco_plant038', 'ara2013_plant091', 'ara2012_plant085', 'ara2013_plant059', 'ara2012_plant030'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['ara2013_plant160', 'ara2012_plant072', 'ara2013_plant101', 'tobacco_plant062', 'ara2013_plant079', 'ara2012_plant017', 'tobacco_plant030', 'ara2013_plant066'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2013_plant011', 'ara2013_plant007', 'tobacco_plant012', 'ara2012_plant111', 'ara2012_plant120', 'ara2012_plant031', 'ara2012_plant047', 'ara2013_plant009'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['ara2013_plant052', 'ara2012_plant119', 'ara2012_plant023', 'ara2012_plant025', 'ara2013_plant090', 'ara2013_plant040', 'ara2013_plant016', 'ara2012_plant110'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2012_plant089', 'ara2012_plant044', 'ara2012_plant039', 'ara2012_plant097', 'tobacco_plant009', 'ara2013_plant051', 'ara2013_plant018', 'ara2012_plant063'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2012_plant028', 'ara2013_plant039', 'ara2013_plant133', 'ara2013_plant006', 'ara2013_plant163', 'ara2013_plant153', 'ara2013_plant035', 'ara2013_plant083'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2013_plant019', 'ara2013_plant012', 'ara2013_plant043', 'ara2012_plant018', 'ara2012_plant033', 'ara2013_plant055', 'ara2013_plant097', 'ara2013_plant161'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2012_plant016', 'ara2012_plant053', 'ara2012_plant099', 'ara2012_plant014', 'ara2013_plant158', 'ara2012_plant115', 'ara2013_plant064', 'ara2013_plant049'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=20 names=['ara2013_plant084', 'ara2013_plant063', 'ara2012_plant105', 'ara2013_plant071', 'ara2013_plant056', 'ara2013_plant110', 'ara2013_plant081', 'ara2013_plant162'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['tobacco_plant045', 'ara2013_plant129', 'ara2013_plant117', 'tobacco_plant029', 'ara2012_plant080', 'ara2013_plant086', 'ara2013_plant008', 'ara2013_plant164'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2013_plant096', 'ara2013_plant070', 'ara2013_plant031', 'ara2012_plant092', 'ara2013_plant137', 'ara2013_plant037', 'ara2012_plant040', 'tobacco_plant041'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2013_plant135', 'ara2012_plant056', 'ara2012_plant066', 'ara2012_plant041', 'ara2013_plant151', 'ara2013_plant157', 'ara2012_plant006', 'tobacco_plant060'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2012_plant069', 'tobacco_plant013', 'ara2013_plant075', 'tobacco_plant014', 'ara2012_plant022', 'ara2012_plant046', 'tobacco_plant019', 'ara2012_plant001'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['ara2013_plant048', 'ara2013_plant053', 'ara2013_plant069', 'ara2012_plant012', 'tobacco_plant051', 'ara2013_plant136', 'ara2013_plant149', 'ara2012_plant036'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2012_plant003', 'ara2013_plant045', 'ara2013_plant116', 'ara2013_plant020', 'ara2013_plant099', 'ara2012_plant021', 'ara2013_plant113', 'ara2012_plant011'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['tobacco_plant055', 'ara2013_plant028', 'tobacco_plant040', 'ara2013_plant023', 'ara2012_plant034', 'ara2013_plant100', 'ara2012_plant054', 'ara2013_plant132'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['tobacco_plant034', 'ara2013_plant108', 'ara2013_plant103', 'ara2012_plant107', 'tobacco_plant039', 'ara2012_plant083', 'tobacco_plant024', 'ara2012_plant109'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['tobacco_plant058', 'ara2013_plant080', 'ara2013_plant001', 'ara2013_plant022', 'tobacco_plant018', 'ara2013_plant104', 'ara2012_plant043', 'ara2013_plant147'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['ara2012_plant095', 'ara2012_plant035', 'ara2012_plant077', 'ara2013_plant094', 'ara2013_plant027', 'ara2012_plant057', 'ara2013_plant010', 'ara2013_plant026'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=34 names=['ara2012_plant055', 'ara2012_plant065', 'ara2013_plant032', 'ara2013_plant128', 'ara2013_plant061', 'ara2012_plant019', 'ara2013_plant013', 'ara2012_plant101'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2013_plant025', 'tobacco_plant010', 'ara2012_plant015', 'ara2013_plant140', 'ara2012_plant106', 'ara2013_plant109', 'ara2013_plant030', 'tobacco_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['ara2012_plant048', 'tobacco_plant042', 'ara2013_plant029', 'tobacco_plant033', 'ara2013_plant150', 'ara2013_plant047', 'ara2013_plant092', 'ara2012_plant114'] — 1048576 elemen.
Epoch  16/100 | Loss: 0.0946 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=0 names=['ara2012_plant055', 'ara2012_plant081', 'ara2012_plant047', 'ara2013_plant089', 'ara2012_plant083', 'tobacco_plant020', 'ara2013_plant030', 'tobacco_plant033'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=1 names=['tobacco_plant015', 'ara2013_plant109', 'ara2013_plant147', 'ara2013_plant128', 'ara2013_plant165', 'ara2012_plant090', 'ara2013_plant110', 'ara2013_plant046'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['ara2012_plant045', 'ara2013_plant080', 'ara2013_plant047', 'ara2013_plant029', 'ara2012_plant017', 'ara2012_plant094', 'ara2012_plant062', 'ara2012_plant089'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['tobacco_plant042', 'ara2013_plant135', 'ara2012_plant029', 'ara2013_plant005', 'tobacco_plant018', 'ara2013_plant071', 'ara2013_plant140', 'ara2012_plant057'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['tobacco_plant060', 'ara2013_plant064', 'ara2013_plant057', 'ara2012_plant012', 'ara2013_plant034', 'ara2013_plant134', 'tobacco_plant041', 'ara2012_plant102'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=8 names=['ara2013_plant083', 'tobacco_plant025', 'ara2013_plant085', 'ara2012_plant030', 'ara2012_plant093', 'ara2013_plant126', 'ara2013_plant035', 'ara2012_plant046'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=9 names=['tobacco_plant061', 'ara2013_plant039', 'ara2012_plant051', 'ara2012_plant059', 'ara2013_plant143', 'tobacco_plant012', 'ara2013_plant001', 'ara2013_plant027'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['ara2013_plant049', 'ara2013_plant114', 'tobacco_plant053', 'ara2013_plant025', 'ara2013_plant014', 'ara2013_plant055', 'ara2013_plant048', 'tobacco_plant008'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2013_plant006', 'ara2013_plant155', 'ara2013_plant028', 'ara2012_plant043', 'ara2013_plant113', 'ara2013_plant042', 'tobacco_plant009', 'ara2012_plant056'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['ara2013_plant002', 'ara2013_plant043', 'ara2012_plant053', 'ara2013_plant051', 'tobacco_plant029', 'ara2013_plant084', 'ara2013_plant163', 'ara2013_plant020'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['ara2013_plant010', 'ara2013_plant037', 'ara2012_plant078', 'ara2012_plant037', 'ara2012_plant035', 'ara2013_plant148', 'ara2012_plant052', 'ara2012_plant054'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2013_plant041', 'tobacco_plant051', 'ara2013_plant026', 'tobacco_plant030', 'ara2013_plant136', 'ara2013_plant141', 'ara2012_plant114', 'ara2013_plant023'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['ara2013_plant153', 'ara2013_plant032', 'ara2013_plant031', 'ara2013_plant106', 'tobacco_plant001', 'ara2012_plant027', 'ara2013_plant096', 'ara2013_plant098'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2012_plant024', 'tobacco_plant022', 'ara2013_plant111', 'ara2012_plant015', 'ara2013_plant076', 'ara2013_plant099', 'ara2013_plant151', 'ara2012_plant074'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2013_plant161', 'ara2012_plant104', 'ara2013_plant062', 'ara2013_plant159', 'ara2012_plant023', 'ara2012_plant119', 'ara2012_plant034', 'ara2012_plant106'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2013_plant115', 'ara2013_plant022', 'tobacco_plant050', 'ara2012_plant021', 'ara2012_plant080', 'ara2013_plant012', 'ara2013_plant097', 'ara2013_plant065'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2013_plant160', 'tobacco_plant040', 'ara2013_plant130', 'tobacco_plant036', 'ara2012_plant110', 'ara2013_plant072', 'ara2012_plant020', 'ara2013_plant103'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['ara2013_plant144', 'ara2013_plant121', 'ara2012_plant033', 'ara2012_plant032', 'ara2012_plant085', 'ara2013_plant094', 'ara2013_plant100', 'ara2013_plant052'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2012_plant116', 'ara2013_plant145', 'ara2013_plant015', 'ara2013_plant063', 'ara2013_plant053', 'ara2012_plant111', 'ara2012_plant118', 'ara2012_plant095'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['tobacco_plant006', 'ara2013_plant105', 'ara2012_plant067', 'tobacco_plant007', 'tobacco_plant010', 'ara2013_plant019', 'ara2013_plant016', 'ara2013_plant154'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['ara2013_plant013', 'ara2012_plant082', 'tobacco_plant039', 'ara2013_plant068', 'tobacco_plant058', 'tobacco_plant059', 'ara2012_plant008', 'ara2013_plant124'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2012_plant013', 'ara2013_plant142', 'ara2012_plant006', 'ara2012_plant071', 'ara2013_plant162', 'ara2012_plant101', 'ara2012_plant076', 'ara2012_plant036'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=36 names=['ara2012_plant077', 'ara2013_plant008', 'ara2013_plant079', 'ara2013_plant021', 'ara2012_plant016', 'ara2013_plant056', 'ara2013_plant102', 'tobacco_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['tobacco_plant023', 'tobacco_plant037', 'ara2012_plant120', 'ara2012_plant031', 'ara2012_plant058', 'ara2013_plant129', 'ara2012_plant009', 'ara2012_plant019'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2013_plant081', 'ara2013_plant018', 'ara2013_plant092', 'ara2013_plant038', 'ara2012_plant018', 'ara2013_plant082', 'tobacco_plant027', 'ara2013_plant070'] — 1048576 elemen.
Epoch  17/100 | Loss: 0.0401 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=0 names=['tobacco_plant006', 'ara2013_plant164', 'ara2013_plant008', 'ara2013_plant090', 'ara2013_plant001', 'tobacco_plant053', 'tobacco_plant052', 'ara2012_plant014'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=3 names=['ara2013_plant118', 'ara2013_plant108', 'ara2012_plant018', 'ara2013_plant089', 'ara2012_plant106', 'ara2013_plant007', 'ara2013_plant041', 'ara2013_plant033'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['tobacco_plant027', 'ara2012_plant030', 'ara2013_plant049', 'ara2012_plant116', 'ara2012_plant078', 'ara2012_plant114', 'ara2013_plant067', 'tobacco_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['tobacco_plant039', 'tobacco_plant025', 'ara2012_plant023', 'ara2013_plant113', 'ara2013_plant137', 'ara2013_plant030', 'ara2012_plant060', 'ara2013_plant110'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2012_plant082', 'ara2013_plant136', 'ara2012_plant033', 'ara2013_plant121', 'ara2013_plant048', 'ara2012_plant036', 'ara2013_plant092', 'ara2012_plant087'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=8 names=['tobacco_plant010', 'tobacco_plant026', 'ara2013_plant037', 'ara2012_plant003', 'tobacco_plant037', 'tobacco_plant051', 'ara2013_plant068', 'ara2012_plant074'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=9 names=['ara2013_plant065', 'ara2013_plant020', 'ara2012_plant069', 'ara2012_plant032', 'ara2012_plant005', 'ara2012_plant019', 'ara2013_plant014', 'ara2013_plant006'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['ara2012_plant057', 'ara2013_plant143', 'ara2013_plant159', 'tobacco_plant038', 'ara2012_plant026', 'tobacco_plant032', 'tobacco_plant035', 'ara2013_plant070'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2013_plant040', 'ara2013_plant149', 'tobacco_plant028', 'ara2012_plant118', 'ara2013_plant046', 'ara2013_plant145', 'ara2012_plant068', 'ara2012_plant073'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2013_plant134', 'ara2013_plant124', 'tobacco_plant042', 'ara2013_plant051', 'ara2012_plant009', 'ara2013_plant013', 'tobacco_plant016', 'ara2012_plant089'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['tobacco_plant045', 'ara2013_plant062', 'tobacco_plant017', 'ara2013_plant005', 'ara2013_plant151', 'ara2013_plant106', 'ara2013_plant141', 'tobacco_plant022'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['ara2012_plant016', 'tobacco_plant014', 'ara2012_plant048', 'ara2012_plant001', 'ara2013_plant132', 'ara2013_plant019', 'tobacco_plant050', 'ara2013_plant058'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2013_plant023', 'ara2012_plant034', 'ara2013_plant018', 'tobacco_plant060', 'ara2013_plant162', 'ara2013_plant144', 'tobacco_plant002', 'ara2013_plant031'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['ara2013_plant102', 'ara2012_plant008', 'ara2012_plant053', 'ara2013_plant117', 'ara2012_plant120', 'tobacco_plant009', 'ara2012_plant055', 'ara2012_plant101'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2013_plant043', 'ara2012_plant117', 'ara2013_plant111', 'ara2012_plant113', 'ara2013_plant063', 'ara2013_plant086', 'ara2013_plant032', 'ara2013_plant099'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=20 names=['tobacco_plant029', 'tobacco_plant044', 'ara2013_plant055', 'ara2013_plant120', 'ara2013_plant021', 'ara2012_plant020', 'ara2013_plant057', 'ara2013_plant073'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['tobacco_plant011', 'tobacco_plant015', 'ara2013_plant114', 'ara2013_plant028', 'tobacco_plant059', 'tobacco_plant024', 'ara2013_plant157', 'ara2013_plant140'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2013_plant069', 'ara2012_plant052', 'ara2013_plant128', 'tobacco_plant012', 'ara2013_plant109', 'tobacco_plant001', 'ara2013_plant116', 'ara2013_plant126'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2013_plant146', 'ara2012_plant039', 'ara2013_plant098', 'ara2012_plant013', 'tobacco_plant004', 'ara2012_plant093', 'tobacco_plant033', 'ara2013_plant027'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2012_plant095', 'ara2013_plant165', 'ara2013_plant084', 'tobacco_plant062', 'ara2013_plant101', 'ara2013_plant044', 'ara2012_plant054', 'ara2013_plant071'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2013_plant138', 'tobacco_plant047', 'ara2013_plant064', 'ara2013_plant075', 'ara2012_plant080', 'ara2013_plant112', 'ara2012_plant109', 'ara2012_plant083'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['ara2012_plant045', 'ara2013_plant009', 'ara2013_plant056', 'ara2012_plant094', 'ara2012_plant115', 'ara2012_plant006', 'tobacco_plant036', 'ara2013_plant010'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['ara2013_plant022', 'ara2013_plant094', 'ara2012_plant081', 'ara2012_plant107', 'tobacco_plant041', 'ara2013_plant139', 'ara2013_plant097', 'ara2013_plant061'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2012_plant111', 'tobacco_plant049', 'ara2012_plant062', 'ara2012_plant119', 'ara2013_plant047', 'ara2012_plant024', 'ara2013_plant133', 'ara2012_plant040'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=32 names=['ara2013_plant156', 'ara2012_plant097', 'ara2012_plant029', 'ara2013_plant078', 'tobacco_plant040', 'ara2013_plant147', 'ara2013_plant015', 'ara2012_plant012'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['ara2013_plant083', 'ara2012_plant011', 'ara2013_plant045', 'ara2012_plant067', 'ara2013_plant096', 'ara2012_plant090', 'ara2012_plant047', 'ara2012_plant076'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=34 names=['ara2012_plant105', 'tobacco_plant018', 'ara2012_plant102', 'ara2013_plant066', 'ara2013_plant163', 'ara2013_plant072', 'ara2013_plant135', 'ara2012_plant075'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=36 names=['ara2012_plant042', 'ara2013_plant081', 'ara2012_plant085', 'ara2012_plant072', 'ara2012_plant046', 'ara2012_plant050', 'ara2013_plant052', 'ara2013_plant127'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2013_plant080', 'ara2012_plant059', 'tobacco_plant013', 'ara2012_plant037', 'ara2012_plant025', 'ara2012_plant100', 'tobacco_plant030', 'ara2012_plant056'] — 1048576 elemen.
Epoch  18/100 | Loss: 0.0378 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=2 names=['tobacco_plant049', 'ara2012_plant033', 'ara2013_plant025', 'ara2012_plant080', 'ara2013_plant155', 'ara2013_plant045', 'ara2012_plant058', 'ara2012_plant097'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['ara2012_plant114', 'ara2013_plant096', 'ara2013_plant063', 'ara2013_plant040', 'ara2013_plant072', 'ara2013_plant120', 'ara2013_plant160', 'tobacco_plant053'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['ara2012_plant045', 'ara2013_plant114', 'ara2013_plant015', 'ara2013_plant151', 'tobacco_plant007', 'ara2012_plant115', 'ara2013_plant071', 'ara2012_plant085'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2012_plant006', 'ara2012_plant113', 'ara2013_plant132', 'ara2013_plant157', 'tobacco_plant037', 'ara2012_plant036', 'tobacco_plant061', 'ara2012_plant073'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['ara2013_plant027', 'ara2012_plant087', 'ara2013_plant055', 'ara2013_plant123', 'ara2012_plant034', 'ara2012_plant026', 'ara2013_plant161', 'ara2013_plant163'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['ara2013_plant038', 'ara2013_plant010', 'ara2012_plant027', 'ara2013_plant125', 'ara2013_plant084', 'ara2013_plant100', 'ara2012_plant013', 'ara2012_plant035'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2013_plant140', 'ara2013_plant104', 'ara2012_plant055', 'ara2013_plant023', 'tobacco_plant041', 'ara2013_plant051', 'ara2013_plant043', 'ara2013_plant121'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2013_plant067', 'ara2013_plant128', 'ara2013_plant143', 'ara2013_plant011', 'tobacco_plant008', 'ara2013_plant034', 'ara2013_plant006', 'ara2013_plant089'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['ara2013_plant146', 'ara2013_plant026', 'ara2013_plant028', 'ara2012_plant051', 'tobacco_plant055', 'ara2012_plant037', 'ara2013_plant076', 'ara2013_plant092'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2013_plant138', 'tobacco_plant002', 'ara2012_plant056', 'ara2012_plant018', 'ara2013_plant086', 'tobacco_plant021', 'ara2013_plant008', 'tobacco_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['ara2013_plant059', 'ara2012_plant014', 'ara2012_plant052', 'tobacco_plant026', 'ara2012_plant012', 'ara2013_plant103', 'ara2013_plant068', 'ara2013_plant036'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2013_plant007', 'ara2012_plant083', 'ara2012_plant021', 'ara2013_plant148', 'ara2013_plant058', 'ara2013_plant117', 'ara2013_plant054', 'ara2013_plant105'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=20 names=['ara2013_plant001', 'tobacco_plant034', 'ara2013_plant141', 'ara2012_plant031', 'ara2013_plant046', 'tobacco_plant059', 'ara2013_plant162', 'ara2012_plant101'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['ara2013_plant154', 'ara2013_plant053', 'ara2013_plant082', 'tobacco_plant048', 'tobacco_plant038', 'ara2013_plant110', 'ara2012_plant001', 'ara2013_plant109'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2013_plant126', 'ara2013_plant080', 'tobacco_plant032', 'ara2013_plant085', 'tobacco_plant020', 'tobacco_plant027', 'tobacco_plant047', 'ara2013_plant165'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2013_plant070', 'ara2013_plant061', 'ara2013_plant137', 'ara2013_plant091', 'ara2012_plant072', 'ara2012_plant047', 'ara2013_plant033', 'ara2012_plant092'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2013_plant108', 'ara2012_plant059', 'ara2012_plant063', 'ara2012_plant065', 'ara2012_plant039', 'ara2013_plant029', 'ara2013_plant113', 'ara2012_plant017'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['tobacco_plant016', 'ara2012_plant105', 'ara2013_plant065', 'ara2012_plant071', 'ara2013_plant020', 'tobacco_plant025', 'ara2012_plant089', 'ara2013_plant044'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2012_plant069', 'ara2012_plant024', 'ara2012_plant103', 'ara2013_plant158', 'tobacco_plant001', 'ara2012_plant081', 'ara2013_plant127', 'ara2012_plant109'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['ara2012_plant116', 'ara2012_plant044', 'ara2013_plant102', 'ara2013_plant115', 'ara2012_plant042', 'ara2013_plant039', 'ara2013_plant164', 'tobacco_plant019'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['ara2012_plant068', 'ara2012_plant020', 'tobacco_plant044', 'ara2013_plant118', 'ara2012_plant019', 'ara2013_plant116', 'ara2012_plant032', 'ara2012_plant111'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2012_plant060', 'ara2012_plant046', 'tobacco_plant013', 'ara2013_plant009', 'ara2013_plant159', 'tobacco_plant058', 'ara2012_plant082', 'ara2012_plant053'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=32 names=['tobacco_plant040', 'ara2013_plant049', 'ara2013_plant144', 'ara2013_plant012', 'ara2013_plant145', 'ara2012_plant011', 'ara2013_plant066', 'ara2013_plant150'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=34 names=['ara2013_plant037', 'ara2013_plant097', 'ara2012_plant008', 'ara2012_plant030', 'ara2013_plant098', 'ara2013_plant134', 'ara2013_plant149', 'ara2013_plant030'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2013_plant048', 'ara2012_plant078', 'ara2013_plant094', 'ara2013_plant042', 'ara2012_plant023', 'tobacco_plant029', 'tobacco_plant006', 'ara2013_plant018'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=36 names=['ara2012_plant104', 'ara2013_plant079', 'ara2012_plant075', 'ara2012_plant028', 'ara2013_plant139', 'ara2013_plant041', 'ara2013_plant047', 'ara2013_plant135'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2013_plant136', 'ara2013_plant002', 'ara2013_plant073', 'ara2012_plant090', 'ara2013_plant081', 'tobacco_plant036', 'tobacco_plant011', 'ara2013_plant016'] — 1048576 elemen.
Epoch  19/100 | Loss: 0.0352 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=0 names=['ara2013_plant025', 'ara2012_plant056', 'tobacco_plant035', 'ara2012_plant011', 'ara2013_plant116', 'ara2012_plant084', 'ara2013_plant092', 'ara2012_plant065'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=1 names=['ara2013_plant102', 'ara2013_plant007', 'tobacco_plant062', 'ara2012_plant088', 'ara2013_plant154', 'tobacco_plant009', 'ara2012_plant093', 'ara2013_plant163'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['ara2013_plant028', 'ara2012_plant031', 'ara2013_plant157', 'ara2013_plant110', 'ara2012_plant077', 'ara2012_plant033', 'ara2012_plant047', 'tobacco_plant051'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=3 names=['ara2012_plant107', 'tobacco_plant059', 'ara2013_plant065', 'tobacco_plant006', 'ara2012_plant089', 'ara2013_plant040', 'ara2013_plant056', 'ara2012_plant073'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2013_plant142', 'ara2013_plant098', 'ara2012_plant092', 'ara2012_plant007', 'ara2012_plant026', 'ara2012_plant099', 'ara2012_plant083', 'ara2013_plant011'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2013_plant135', 'ara2012_plant072', 'ara2012_plant087', 'ara2013_plant042', 'ara2013_plant114', 'ara2013_plant123', 'ara2013_plant055', 'ara2012_plant001'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=8 names=['ara2012_plant053', 'ara2013_plant008', 'ara2012_plant105', 'ara2013_plant058', 'ara2013_plant159', 'ara2013_plant164', 'ara2012_plant094', 'tobacco_plant020'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=9 names=['tobacco_plant013', 'ara2013_plant041', 'ara2013_plant053', 'ara2013_plant080', 'ara2013_plant165', 'tobacco_plant039', 'ara2013_plant072', 'ara2013_plant035'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['tobacco_plant045', 'ara2012_plant039', 'ara2012_plant111', 'ara2013_plant130', 'ara2013_plant086', 'ara2013_plant034', 'ara2012_plant080', 'ara2013_plant063'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['ara2012_plant020', 'ara2012_plant120', 'tobacco_plant024', 'tobacco_plant001', 'ara2013_plant021', 'ara2012_plant037', 'ara2013_plant099', 'ara2013_plant133'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2012_plant021', 'ara2012_plant082', 'ara2012_plant019', 'ara2013_plant047', 'ara2013_plant132', 'ara2013_plant112', 'ara2013_plant070', 'ara2013_plant052'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['tobacco_plant021', 'ara2013_plant014', 'ara2012_plant057', 'ara2013_plant117', 'ara2012_plant030', 'tobacco_plant052', 'tobacco_plant029', 'ara2012_plant024'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['tobacco_plant007', 'ara2013_plant026', 'ara2012_plant075', 'ara2013_plant002', 'ara2013_plant143', 'ara2012_plant042', 'ara2013_plant162', 'tobacco_plant050'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['ara2013_plant089', 'ara2013_plant012', 'ara2013_plant064', 'ara2012_plant103', 'ara2012_plant013', 'ara2012_plant003', 'ara2013_plant019', 'ara2013_plant069'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['tobacco_plant060', 'tobacco_plant044', 'tobacco_plant027', 'ara2013_plant078', 'ara2012_plant034', 'ara2013_plant022', 'ara2013_plant082', 'ara2013_plant057'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2012_plant032', 'ara2013_plant043', 'tobacco_plant055', 'tobacco_plant011', 'ara2013_plant054', 'ara2013_plant062', 'ara2013_plant128', 'ara2012_plant017'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['tobacco_plant049', 'ara2013_plant013', 'ara2013_plant134', 'ara2012_plant014', 'ara2012_plant097', 'ara2013_plant126', 'tobacco_plant032', 'ara2013_plant115'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['ara2013_plant144', 'ara2012_plant109', 'tobacco_plant040', 'ara2013_plant109', 'tobacco_plant017', 'ara2012_plant051', 'ara2013_plant155', 'ara2013_plant071'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2013_plant118', 'ara2013_plant005', 'ara2012_plant085', 'ara2013_plant068', 'tobacco_plant023', 'ara2013_plant108', 'ara2012_plant054', 'ara2013_plant153'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['ara2013_plant039', 'ara2012_plant004', 'ara2013_plant029', 'tobacco_plant046', 'ara2012_plant117', 'ara2013_plant032', 'ara2013_plant136', 'ara2012_plant027'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['tobacco_plant008', 'ara2013_plant111', 'ara2013_plant076', 'ara2012_plant071', 'ara2013_plant140', 'ara2012_plant041', 'ara2012_plant006', 'ara2013_plant150'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['ara2013_plant020', 'ara2012_plant035', 'ara2013_plant113', 'tobacco_plant033', 'ara2013_plant066', 'ara2013_plant090', 'ara2012_plant101', 'ara2013_plant073'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2013_plant016', 'ara2013_plant037', 'ara2012_plant023', 'ara2012_plant069', 'tobacco_plant036', 'ara2013_plant139', 'ara2013_plant158', 'ara2012_plant012'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['ara2013_plant151', 'tobacco_plant010', 'ara2012_plant045', 'tobacco_plant005', 'ara2012_plant113', 'ara2012_plant043', 'tobacco_plant014', 'ara2012_plant102'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['ara2012_plant066', 'ara2012_plant008', 'ara2013_plant033', 'ara2013_plant083', 'ara2013_plant145', 'ara2012_plant055', 'ara2012_plant108', 'ara2013_plant096'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2012_plant005', 'ara2013_plant127', 'ara2013_plant048', 'ara2012_plant074', 'ara2012_plant090', 'ara2012_plant106', 'tobacco_plant004', 'ara2012_plant036'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['ara2013_plant030', 'ara2013_plant075', 'tobacco_plant030', 'ara2013_plant015', 'ara2012_plant028', 'ara2013_plant094', 'ara2012_plant052', 'ara2012_plant029'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=34 names=['ara2012_plant062', 'ara2013_plant031', 'ara2013_plant105', 'ara2013_plant061', 'tobacco_plant038', 'ara2013_plant160', 'ara2012_plant078', 'ara2013_plant010'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2013_plant084', 'ara2013_plant137', 'tobacco_plant025', 'ara2013_plant036', 'ara2013_plant067', 'ara2013_plant081', 'ara2012_plant100', 'ara2013_plant121'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2012_plant009', 'ara2012_plant110', 'tobacco_plant034', 'ara2013_plant027', 'ara2012_plant116', 'tobacco_plant012', 'ara2013_plant129', 'ara2013_plant045'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=39 names=['tobacco_plant061'] — 131072 elemen.
Epoch  20/100 | Loss: 0.0393 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=1 names=['ara2012_plant008', 'ara2013_plant070', 'ara2013_plant016', 'tobacco_plant035', 'ara2013_plant081', 'ara2013_plant159', 'ara2013_plant110', 'ara2013_plant046'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['ara2013_plant103', 'ara2013_plant082', 'tobacco_plant028', 'ara2013_plant137', 'ara2013_plant039', 'tobacco_plant019', 'ara2013_plant128', 'ara2013_plant057'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['ara2012_plant020', 'ara2013_plant149', 'ara2012_plant041', 'ara2013_plant022', 'ara2013_plant135', 'tobacco_plant051', 'ara2013_plant086', 'ara2013_plant041'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2012_plant048', 'ara2012_plant110', 'ara2013_plant160', 'ara2012_plant089', 'ara2012_plant033', 'tobacco_plant053', 'ara2012_plant024', 'ara2013_plant144'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['tobacco_plant032', 'ara2013_plant115', 'tobacco_plant039', 'ara2012_plant056', 'ara2013_plant049', 'ara2012_plant001', 'ara2012_plant087', 'ara2012_plant002'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=9 names=['ara2013_plant067', 'ara2013_plant071', 'ara2012_plant073', 'ara2013_plant038', 'ara2013_plant075', 'ara2013_plant139', 'ara2013_plant034', 'ara2013_plant098'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['ara2012_plant092', 'ara2013_plant029', 'ara2013_plant120', 'ara2013_plant062', 'ara2012_plant106', 'tobacco_plant026', 'ara2013_plant097', 'ara2013_plant076'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['tobacco_plant040', 'ara2013_plant027', 'ara2012_plant017', 'ara2012_plant051', 'ara2013_plant084', 'ara2013_plant010', 'ara2012_plant045', 'ara2012_plant059'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2013_plant020', 'ara2012_plant036', 'ara2012_plant027', 'ara2013_plant080', 'ara2013_plant156', 'ara2012_plant022', 'ara2012_plant120', 'ara2013_plant008'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['ara2012_plant046', 'ara2013_plant023', 'ara2012_plant062', 'ara2012_plant117', 'ara2012_plant028', 'ara2012_plant026', 'tobacco_plant041', 'ara2013_plant117'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['tobacco_plant029', 'ara2012_plant109', 'ara2013_plant032', 'tobacco_plant045', 'ara2013_plant072', 'ara2012_plant114', 'ara2012_plant076', 'ara2013_plant164'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2012_plant047', 'tobacco_plant037', 'ara2012_plant057', 'ara2012_plant100', 'ara2013_plant118', 'ara2013_plant021', 'ara2012_plant052', 'ara2012_plant107'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2012_plant021', 'ara2012_plant097', 'ara2013_plant061', 'ara2012_plant116', 'ara2013_plant045', 'ara2013_plant007', 'ara2013_plant042', 'tobacco_plant046'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2012_plant058', 'ara2013_plant141', 'ara2012_plant006', 'tobacco_plant050', 'ara2012_plant040', 'tobacco_plant017', 'ara2013_plant133', 'ara2012_plant099'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['ara2013_plant069', 'ara2013_plant025', 'ara2012_plant060', 'ara2013_plant114', 'ara2013_plant011', 'ara2012_plant055', 'ara2012_plant037', 'ara2012_plant015'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2012_plant044', 'tobacco_plant018', 'ara2013_plant033', 'ara2012_plant035', 'tobacco_plant059', 'tobacco_plant038', 'ara2012_plant029', 'ara2013_plant006'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2012_plant078', 'ara2012_plant054', 'ara2013_plant036', 'ara2013_plant153', 'ara2013_plant059', 'ara2012_plant071', 'ara2012_plant088', 'ara2012_plant012'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['ara2013_plant037', 'ara2013_plant048', 'ara2012_plant074', 'tobacco_plant025', 'tobacco_plant030', 'ara2012_plant016', 'ara2013_plant111', 'tobacco_plant008'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['tobacco_plant036', 'ara2012_plant119', 'tobacco_plant048', 'tobacco_plant020', 'ara2013_plant102', 'ara2013_plant031', 'ara2013_plant001', 'tobacco_plant004'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2012_plant042', 'ara2013_plant053', 'ara2013_plant161', 'ara2013_plant018', 'ara2012_plant108', 'ara2012_plant111', 'ara2013_plant104', 'ara2013_plant091'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['tobacco_plant001', 'ara2013_plant002', 'ara2013_plant130', 'ara2012_plant082', 'ara2012_plant068', 'ara2012_plant066', 'ara2013_plant014', 'ara2012_plant043'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['ara2013_plant127', 'tobacco_plant062', 'tobacco_plant034', 'ara2013_plant154', 'ara2013_plant113', 'ara2012_plant005', 'ara2013_plant155', 'ara2013_plant064'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2013_plant140', 'ara2012_plant104', 'ara2012_plant105', 'ara2013_plant009', 'ara2012_plant118', 'ara2012_plant083', 'ara2013_plant129', 'ara2013_plant145'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=36 names=['ara2013_plant092', 'ara2012_plant085', 'ara2013_plant047', 'ara2013_plant028', 'ara2013_plant085', 'ara2013_plant019', 'ara2013_plant013', 'ara2012_plant019'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['ara2013_plant157', 'ara2013_plant055', 'ara2013_plant163', 'ara2013_plant116', 'ara2013_plant094', 'tobacco_plant002', 'ara2012_plant095', 'tobacco_plant005'] — 1048576 elemen.
Epoch  21/100 | Loss: 0.0406 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=1 names=['ara2013_plant158', 'ara2013_plant146', 'ara2012_plant044', 'ara2013_plant120', 'ara2013_plant129', 'ara2013_plant159', 'ara2013_plant132', 'ara2012_plant034'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['tobacco_plant023', 'ara2012_plant013', 'ara2012_plant017', 'ara2012_plant089', 'ara2012_plant094', 'ara2013_plant008', 'ara2013_plant037', 'ara2012_plant023'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=3 names=['ara2012_plant015', 'ara2013_plant124', 'ara2012_plant066', 'ara2012_plant093', 'ara2012_plant059', 'ara2012_plant071', 'ara2013_plant116', 'ara2012_plant069'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['ara2012_plant116', 'ara2013_plant098', 'ara2012_plant011', 'tobacco_plant036', 'ara2013_plant102', 'ara2012_plant028', 'ara2013_plant085', 'ara2013_plant068'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2013_plant154', 'ara2012_plant007', 'ara2013_plant163', 'tobacco_plant034', 'ara2012_plant053', 'ara2012_plant008', 'ara2012_plant108', 'ara2013_plant069'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['ara2012_plant101', 'ara2012_plant072', 'ara2012_plant042', 'ara2013_plant023', 'tobacco_plant033', 'ara2012_plant002', 'ara2012_plant083', 'ara2013_plant125'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2013_plant040', 'ara2013_plant123', 'ara2012_plant119', 'ara2012_plant077', 'tobacco_plant029', 'ara2012_plant033', 'ara2012_plant006', 'ara2012_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=9 names=['tobacco_plant026', 'tobacco_plant062', 'ara2013_plant139', 'ara2013_plant041', 'tobacco_plant010', 'ara2012_plant039', 'tobacco_plant038', 'ara2013_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['ara2013_plant002', 'ara2012_plant030', 'tobacco_plant053', 'ara2013_plant073', 'ara2012_plant020', 'tobacco_plant021', 'ara2012_plant041', 'ara2013_plant147'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['ara2012_plant018', 'ara2012_plant055', 'ara2013_plant027', 'ara2013_plant007', 'ara2013_plant084', 'ara2013_plant012', 'ara2013_plant134', 'ara2012_plant114'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2013_plant148', 'ara2013_plant033', 'ara2013_plant140', 'ara2012_plant113', 'ara2013_plant044', 'tobacco_plant005', 'ara2013_plant022', 'ara2012_plant001'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['tobacco_plant058', 'tobacco_plant028', 'tobacco_plant040', 'tobacco_plant001', 'tobacco_plant032', 'tobacco_plant007', 'ara2012_plant045', 'ara2013_plant015'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['ara2013_plant047', 'ara2013_plant020', 'ara2012_plant048', 'ara2013_plant157', 'tobacco_plant035', 'tobacco_plant027', 'ara2013_plant010', 'ara2013_plant110'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2013_plant083', 'ara2012_plant057', 'ara2013_plant019', 'ara2012_plant109', 'tobacco_plant060', 'ara2012_plant035', 'ara2013_plant164', 'tobacco_plant009'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['ara2013_plant103', 'ara2012_plant095', 'ara2012_plant080', 'ara2013_plant086', 'ara2013_plant100', 'ara2013_plant096', 'ara2013_plant014', 'ara2012_plant120'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2012_plant029', 'ara2013_plant149', 'ara2012_plant036', 'tobacco_plant061', 'ara2013_plant160', 'ara2013_plant151', 'ara2012_plant016', 'ara2013_plant016'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['tobacco_plant017', 'ara2013_plant031', 'ara2013_plant067', 'ara2013_plant051', 'ara2013_plant114', 'tobacco_plant019', 'ara2012_plant107', 'tobacco_plant045'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['tobacco_plant052', 'ara2013_plant075', 'ara2013_plant045', 'ara2013_plant127', 'ara2013_plant076', 'ara2013_plant097', 'tobacco_plant048', 'ara2013_plant156'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['tobacco_plant006', 'ara2013_plant026', 'ara2012_plant032', 'ara2013_plant009', 'ara2013_plant059', 'ara2012_plant043', 'ara2012_plant075', 'ara2013_plant133'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2012_plant003', 'ara2012_plant102', 'tobacco_plant008', 'ara2013_plant001', 'tobacco_plant049', 'ara2013_plant135', 'ara2013_plant105', 'ara2012_plant009'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2012_plant019', 'tobacco_plant015', 'ara2013_plant137', 'ara2012_plant051', 'ara2013_plant030', 'ara2013_plant080', 'ara2013_plant034', 'ara2013_plant142'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['ara2012_plant062', 'ara2012_plant088', 'ara2013_plant053', 'ara2013_plant104', 'ara2012_plant085', 'ara2012_plant058', 'ara2013_plant106', 'ara2012_plant081'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['tobacco_plant012', 'ara2012_plant063', 'ara2012_plant092', 'ara2013_plant094', 'ara2013_plant101', 'ara2012_plant054', 'ara2013_plant162', 'ara2012_plant046'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['ara2013_plant126', 'ara2013_plant046', 'ara2013_plant025', 'ara2013_plant070', 'ara2013_plant064', 'tobacco_plant039', 'ara2013_plant117', 'ara2012_plant117'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['ara2013_plant013', 'ara2013_plant145', 'ara2013_plant038', 'ara2013_plant089', 'ara2012_plant104', 'ara2012_plant111', 'ara2012_plant060', 'ara2012_plant014'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['tobacco_plant042', 'tobacco_plant046', 'ara2013_plant144', 'ara2013_plant112', 'ara2013_plant118', 'ara2013_plant054', 'tobacco_plant018', 'ara2013_plant036'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2013_plant143', 'tobacco_plant004', 'ara2013_plant062', 'tobacco_plant024', 'ara2013_plant057', 'ara2013_plant161', 'ara2012_plant021', 'ara2013_plant081'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=34 names=['ara2012_plant027', 'ara2013_plant011', 'ara2012_plant099', 'ara2013_plant153', 'ara2013_plant028', 'ara2013_plant141', 'ara2013_plant058', 'ara2012_plant087'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2013_plant061', 'tobacco_plant013', 'tobacco_plant014', 'ara2013_plant108', 'ara2013_plant049', 'tobacco_plant037', 'ara2013_plant079', 'tobacco_plant016'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2012_plant115', 'tobacco_plant041', 'ara2013_plant032', 'ara2012_plant073', 'ara2012_plant012', 'ara2013_plant021', 'ara2013_plant071', 'ara2013_plant048'] — 1048576 elemen.
Epoch  22/100 | Loss: 0.0395 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=0 names=['ara2013_plant153', 'ara2013_plant047', 'ara2013_plant134', 'tobacco_plant023', 'ara2012_plant059', 'ara2013_plant092', 'tobacco_plant053', 'tobacco_plant002'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=1 names=['ara2013_plant139', 'ara2013_plant127', 'ara2013_plant147', 'ara2012_plant119', 'ara2013_plant099', 'ara2013_plant052', 'ara2012_plant052', 'ara2013_plant161'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2012_plant013', 'ara2013_plant114', 'ara2012_plant106', 'ara2012_plant077', 'tobacco_plant036', 'ara2013_plant068', 'ara2013_plant164', 'tobacco_plant014'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['tobacco_plant046', 'ara2012_plant080', 'ara2012_plant005', 'tobacco_plant032', 'ara2013_plant091', 'ara2013_plant032', 'ara2013_plant141', 'ara2012_plant015'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2013_plant059', 'ara2013_plant029', 'ara2013_plant030', 'ara2012_plant037', 'ara2012_plant116', 'tobacco_plant042', 'ara2013_plant159', 'ara2012_plant031'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=9 names=['ara2013_plant121', 'ara2012_plant087', 'ara2013_plant156', 'ara2012_plant105', 'ara2012_plant002', 'tobacco_plant015', 'ara2012_plant110', 'ara2013_plant096'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2013_plant071', 'tobacco_plant048', 'ara2013_plant034', 'ara2012_plant028', 'ara2012_plant104', 'tobacco_plant033', 'ara2012_plant056', 'ara2013_plant078'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2012_plant101', 'ara2012_plant023', 'ara2012_plant032', 'ara2013_plant036', 'ara2013_plant158', 'tobacco_plant005', 'ara2013_plant049', 'ara2012_plant004'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['tobacco_plant038', 'ara2013_plant163', 'ara2012_plant018', 'ara2012_plant109', 'tobacco_plant012', 'ara2013_plant055', 'ara2013_plant084', 'ara2013_plant066'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['tobacco_plant006', 'ara2012_plant041', 'ara2013_plant094', 'ara2013_plant010', 'ara2013_plant075', 'ara2013_plant142', 'ara2012_plant065', 'ara2012_plant030'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2013_plant112', 'ara2013_plant151', 'ara2012_plant060', 'ara2012_plant054', 'ara2013_plant154', 'ara2013_plant046', 'tobacco_plant026', 'ara2013_plant117'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2013_plant086', 'ara2012_plant076', 'tobacco_plant024', 'ara2013_plant064', 'ara2012_plant083', 'ara2013_plant155', 'ara2013_plant011', 'ara2013_plant108'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=20 names=['ara2013_plant070', 'ara2012_plant097', 'ara2012_plant051', 'ara2013_plant001', 'tobacco_plant039', 'ara2013_plant137', 'ara2013_plant150', 'ara2012_plant095'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['ara2013_plant072', 'ara2012_plant114', 'tobacco_plant008', 'ara2013_plant002', 'ara2012_plant003', 'ara2013_plant019', 'ara2013_plant133', 'ara2013_plant048'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['tobacco_plant035', 'tobacco_plant011', 'ara2012_plant042', 'tobacco_plant049', 'ara2013_plant148', 'ara2012_plant102', 'ara2012_plant040', 'ara2013_plant105'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['ara2013_plant109', 'ara2013_plant090', 'ara2012_plant020', 'ara2013_plant031', 'ara2012_plant099', 'ara2012_plant022', 'ara2013_plant067', 'ara2013_plant053'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['tobacco_plant040', 'ara2013_plant079', 'ara2013_plant138', 'ara2013_plant120', 'ara2012_plant117', 'ara2013_plant054', 'tobacco_plant029', 'ara2012_plant012'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['ara2012_plant055', 'tobacco_plant019', 'ara2013_plant145', 'ara2012_plant006', 'tobacco_plant051', 'ara2013_plant128', 'tobacco_plant037', 'ara2013_plant143'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2013_plant041', 'tobacco_plant055', 'ara2013_plant033', 'ara2012_plant072', 'ara2013_plant012', 'ara2013_plant126', 'ara2012_plant029', 'ara2013_plant015'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['ara2012_plant082', 'ara2012_plant050', 'ara2012_plant021', 'ara2012_plant108', 'ara2013_plant025', 'ara2013_plant038', 'tobacco_plant022', 'ara2013_plant100'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['ara2013_plant135', 'ara2012_plant025', 'ara2012_plant113', 'ara2012_plant089', 'ara2012_plant107', 'ara2013_plant013', 'ara2013_plant007', 'ara2012_plant068'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2013_plant160', 'ara2013_plant085', 'tobacco_plant060', 'ara2012_plant045', 'ara2013_plant140', 'ara2012_plant039', 'ara2013_plant125', 'ara2013_plant057'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=32 names=['tobacco_plant010', 'ara2012_plant078', 'tobacco_plant030', 'ara2013_plant063', 'ara2013_plant062', 'ara2013_plant157', 'tobacco_plant021', 'ara2013_plant018'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['ara2012_plant011', 'ara2012_plant071', 'ara2012_plant033', 'tobacco_plant058', 'tobacco_plant004', 'ara2013_plant042', 'ara2013_plant039', 'tobacco_plant007'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=34 names=['ara2013_plant103', 'ara2013_plant009', 'ara2013_plant051', 'ara2013_plant021', 'tobacco_plant034', 'ara2012_plant066', 'ara2013_plant162', 'ara2012_plant120'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['tobacco_plant009', 'ara2013_plant081', 'ara2013_plant061', 'ara2013_plant065', 'ara2012_plant001', 'ara2012_plant014', 'ara2012_plant034', 'ara2013_plant083'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['ara2013_plant101', 'ara2012_plant084', 'ara2012_plant047', 'ara2013_plant076', 'ara2012_plant073', 'ara2012_plant058', 'ara2013_plant132', 'ara2012_plant057'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2013_plant113', 'ara2013_plant104', 'ara2013_plant040', 'ara2013_plant044', 'ara2013_plant069', 'ara2012_plant115', 'tobacco_plant041', 'ara2012_plant009'] — 1048576 elemen.
Epoch  23/100 | Loss: 0.0629 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=0 names=['tobacco_plant018', 'ara2012_plant025', 'ara2012_plant109', 'ara2012_plant075', 'ara2012_plant108', 'ara2012_plant084', 'ara2013_plant089', 'tobacco_plant034'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['ara2013_plant143', 'ara2013_plant010', 'ara2012_plant085', 'ara2013_plant030', 'tobacco_plant022', 'ara2013_plant080', 'ara2013_plant026', 'tobacco_plant052'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=3 names=['ara2012_plant068', 'ara2012_plant093', 'ara2012_plant081', 'ara2013_plant046', 'ara2012_plant107', 'ara2012_plant041', 'ara2013_plant103', 'ara2013_plant135'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['ara2013_plant052', 'ara2012_plant088', 'ara2013_plant150', 'ara2012_plant003', 'tobacco_plant047', 'ara2012_plant045', 'ara2012_plant073', 'ara2013_plant148'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2013_plant016', 'ara2013_plant067', 'ara2013_plant042', 'ara2013_plant162', 'ara2013_plant023', 'ara2013_plant090', 'ara2013_plant069', 'ara2013_plant163'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['ara2013_plant047', 'ara2013_plant009', 'ara2012_plant095', 'ara2013_plant144', 'ara2013_plant137', 'ara2013_plant084', 'ara2013_plant070', 'ara2012_plant113'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2013_plant014', 'ara2013_plant102', 'tobacco_plant002', 'ara2013_plant051', 'ara2013_plant029', 'tobacco_plant001', 'ara2013_plant006', 'ara2012_plant033'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['ara2013_plant012', 'tobacco_plant017', 'ara2012_plant027', 'ara2013_plant039', 'ara2012_plant069', 'ara2012_plant019', 'ara2012_plant030', 'ara2013_plant054'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['tobacco_plant046', 'ara2013_plant001', 'ara2013_plant134', 'ara2012_plant048', 'ara2013_plant008', 'ara2013_plant055', 'tobacco_plant015', 'ara2013_plant059'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['tobacco_plant048', 'ara2013_plant128', 'ara2012_plant001', 'ara2013_plant040', 'ara2012_plant117', 'ara2013_plant085', 'ara2013_plant079', 'ara2012_plant065'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2013_plant036', 'ara2013_plant151', 'ara2012_plant106', 'ara2013_plant063', 'ara2012_plant060', 'ara2012_plant028', 'ara2013_plant086', 'ara2012_plant090'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['ara2012_plant083', 'ara2013_plant027', 'tobacco_plant029', 'ara2013_plant049', 'ara2013_plant072', 'tobacco_plant013', 'ara2013_plant127', 'ara2012_plant056'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2012_plant051', 'ara2012_plant089', 'ara2012_plant087', 'ara2013_plant110', 'ara2013_plant096', 'ara2012_plant099', 'ara2013_plant121', 'ara2012_plant037'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2012_plant002', 'tobacco_plant049', 'ara2012_plant120', 'ara2012_plant016', 'ara2013_plant116', 'ara2013_plant007', 'ara2012_plant024', 'ara2012_plant007'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=20 names=['ara2012_plant032', 'ara2012_plant063', 'ara2012_plant006', 'ara2013_plant064', 'ara2013_plant142', 'ara2013_plant141', 'ara2012_plant114', 'ara2012_plant097'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['tobacco_plant010', 'ara2013_plant028', 'ara2013_plant120', 'ara2012_plant066', 'ara2012_plant054', 'ara2013_plant099', 'tobacco_plant006', 'tobacco_plant038'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['ara2013_plant033', 'ara2012_plant104', 'ara2012_plant116', 'ara2013_plant140', 'ara2013_plant020', 'tobacco_plant042', 'ara2013_plant106', 'ara2013_plant126'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2013_plant015', 'ara2013_plant124', 'ara2012_plant017', 'ara2013_plant094', 'tobacco_plant005', 'tobacco_plant014', 'ara2012_plant009', 'ara2012_plant020'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2013_plant109', 'ara2012_plant071', 'ara2012_plant034', 'ara2013_plant098', 'tobacco_plant011', 'ara2013_plant031', 'ara2013_plant066', 'ara2013_plant092'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['ara2013_plant038', 'tobacco_plant023', 'ara2013_plant061', 'ara2013_plant021', 'tobacco_plant055', 'tobacco_plant007', 'ara2013_plant057', 'tobacco_plant021'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['tobacco_plant035', 'ara2012_plant015', 'tobacco_plant016', 'ara2012_plant072', 'ara2013_plant062', 'ara2013_plant013', 'ara2013_plant043', 'ara2013_plant133'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['ara2013_plant100', 'ara2013_plant129', 'ara2012_plant043', 'ara2013_plant147', 'ara2012_plant092', 'tobacco_plant027', 'ara2013_plant068', 'ara2013_plant019'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['tobacco_plant009', 'ara2013_plant041', 'ara2012_plant031', 'ara2012_plant040', 'tobacco_plant036', 'ara2013_plant011', 'tobacco_plant039', 'tobacco_plant059'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['ara2013_plant044', 'ara2013_plant037', 'tobacco_plant040', 'tobacco_plant028', 'ara2012_plant022', 'ara2013_plant155', 'tobacco_plant062', 'ara2012_plant074'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['ara2013_plant076', 'ara2012_plant062', 'ara2013_plant130', 'ara2012_plant115', 'ara2012_plant011', 'tobacco_plant004', 'tobacco_plant060', 'ara2013_plant158'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2012_plant102', 'tobacco_plant051', 'ara2012_plant013', 'ara2013_plant153', 'ara2013_plant097', 'ara2013_plant132', 'ara2012_plant042', 'ara2012_plant018'] — 1048576 elemen.
Epoch  24/100 | Loss: 0.0383 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
Early stopping at epoch 24
  ✓ Saved: best_standard_bce.pth (124.2 MB)
  ✓ Saved: config_standard_bce.yaml
  ✓ Saved: metrics_standard_bce.json

============================================================
EXPERIMENT: weighted_bce_dice
============================================================
Dataset: 347 samples, binary (bg + leaf)
Train: 313  Val: 34
[wandb] init failed (init() got an unexpected keyword argument 'finish_previous') — continuing without tracking
Epochs: 100, Early stop patience: 15, Delta: 0.005
------------------------------------------------------------
Epoch   1/100 | Loss: 0.4268 | Val Loss: 50.1264 | IoU: 0.0005 | Dice: 0.0010
  ★ New best Dice: 0.0010
Epoch   2/100 | Loss: 0.1925 | Val Loss: 3.5419 | IoU: 0.0755 | Dice: 0.1395
  ★ New best Dice: 0.1395
Epoch   3/100 | Loss: 0.1421 | Val Loss: 6.7098 | IoU: 0.4122 | Dice: 0.5571
  ★ New best Dice: 0.5571
Epoch   4/100 | Loss: 0.1424 | Val Loss: 0.1590 | IoU: 0.9067 | Dice: 0.9510
  ★ New best Dice: 0.9510
Epoch   5/100 | Loss: 0.1310 | Val Loss: 0.8301 | IoU: 0.7656 | Dice: 0.8661
Epoch   6/100 | Loss: 0.1174 | Val Loss: 0.3509 | IoU: 0.8642 | Dice: 0.9272
Epoch   7/100 | Loss: 0.1086 | Val Loss: 0.7439 | IoU: 0.7760 | Dice: 0.8721
Epoch   8/100 | Loss: 0.1048 | Val Loss: 0.3293 | IoU: 0.8268 | Dice: 0.9047
Epoch   9/100 | Loss: 0.0987 | Val Loss: 0.1963 | IoU: 0.8980 | Dice: 0.9462
Epoch  10/100 | Loss: 0.1008 | Val Loss: 0.1276 | IoU: 0.9156 | Dice: 0.9558
[WARN (C)] pred NON-FINITE at bi=29 names=['ara2013_plant041', 'tobacco_plant059', 'ara2013_plant067', 'ara2013_plant064', 'ara2013_plant147', 'tobacco_plant055', 'ara2012_plant109', 'ara2013_plant084'] — 1048576 elemen.
[WARN (B)] grad NaN at bi=39 names=['tobacco_plant007'] — step di-skip.
[INFO] Epoch 11: 1 batch di-skip karena NaN/Inf (diagnosis B/C).
Epoch  11/100 | Loss: 0.1018 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (D)] IoU anjlok drastis: 0.9156 → 0.0000 di epoch 11. Cek prediksi di results/weighted_bce_dice/predictions.
[WARN (C)] pred NON-FINITE at bi=3 names=['tobacco_plant029', 'ara2013_plant005', 'ara2013_plant156', 'ara2012_plant027', 'ara2013_plant102', 'ara2013_plant026', 'ara2013_plant018', 'tobacco_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['ara2013_plant051', 'ara2013_plant159', 'ara2013_plant071', 'ara2013_plant109', 'ara2013_plant012', 'ara2013_plant101', 'ara2012_plant097', 'ara2013_plant002'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2013_plant061', 'ara2012_plant101', 'tobacco_plant034', 'ara2013_plant097', 'ara2013_plant165', 'ara2012_plant047', 'tobacco_plant018', 'ara2013_plant104'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['tobacco_plant021', 'ara2013_plant041', 'ara2013_plant072', 'ara2013_plant144', 'ara2013_plant096', 'ara2013_plant068', 'ara2012_plant114', 'tobacco_plant030'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['ara2013_plant020', 'tobacco_plant017', 'ara2012_plant095', 'ara2012_plant087', 'ara2012_plant084', 'ara2013_plant142', 'ara2012_plant001', 'ara2012_plant014'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2012_plant102', 'ara2012_plant013', 'ara2012_plant003', 'ara2013_plant155', 'ara2013_plant153', 'ara2013_plant059', 'ara2013_plant035', 'ara2013_plant011'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=20 names=['ara2012_plant024', 'ara2013_plant052', 'ara2013_plant069', 'ara2013_plant112', 'tobacco_plant044', 'tobacco_plant060', 'ara2013_plant008', 'tobacco_plant035'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['ara2012_plant007', 'ara2012_plant032', 'ara2012_plant008', 'ara2012_plant043', 'ara2013_plant114', 'ara2012_plant073', 'ara2013_plant083', 'ara2013_plant042'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['ara2013_plant130', 'tobacco_plant038', 'ara2013_plant062', 'ara2013_plant158', 'ara2013_plant044', 'ara2013_plant076', 'ara2013_plant134', 'ara2012_plant025'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2012_plant078', 'ara2012_plant028', 'ara2012_plant106', 'ara2013_plant081', 'ara2012_plant060', 'ara2013_plant023', 'ara2013_plant033', 'ara2012_plant056'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2013_plant066', 'ara2012_plant117', 'ara2012_plant104', 'ara2013_plant007', 'tobacco_plant058', 'ara2013_plant128', 'ara2013_plant164', 'tobacco_plant039'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2013_plant010', 'ara2013_plant145', 'ara2013_plant016', 'ara2013_plant116', 'tobacco_plant015', 'ara2012_plant046', 'tobacco_plant062', 'ara2013_plant075'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=36 names=['ara2013_plant067', 'ara2013_plant063', 'tobacco_plant014', 'ara2012_plant016', 'ara2012_plant094', 'ara2012_plant053', 'ara2013_plant139', 'ara2012_plant042'] — 1048576 elemen.
Epoch  12/100 | Loss: 0.1003 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=0 names=['tobacco_plant007', 'ara2012_plant076', 'ara2013_plant005', 'ara2013_plant045', 'ara2013_plant096', 'ara2012_plant029', 'ara2013_plant011', 'ara2013_plant111'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['tobacco_plant040', 'ara2012_plant017', 'ara2013_plant101', 'ara2013_plant021', 'tobacco_plant021', 'ara2013_plant157', 'ara2012_plant044', 'ara2013_plant120'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=3 names=['ara2013_plant063', 'ara2013_plant150', 'ara2012_plant067', 'ara2013_plant046', 'ara2012_plant055', 'ara2013_plant114', 'ara2012_plant020', 'tobacco_plant044'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2013_plant153', 'ara2013_plant128', 'tobacco_plant023', 'ara2013_plant014', 'tobacco_plant011', 'ara2012_plant108', 'ara2013_plant126', 'ara2012_plant104'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['ara2012_plant106', 'ara2012_plant118', 'ara2013_plant059', 'ara2012_plant082', 'tobacco_plant002', 'ara2012_plant110', 'ara2013_plant049', 'ara2012_plant042'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2012_plant095', 'ara2013_plant006', 'ara2013_plant103', 'ara2012_plant026', 'ara2012_plant090', 'ara2012_plant120', 'ara2012_plant034', 'ara2013_plant002'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=8 names=['ara2012_plant094', 'ara2013_plant145', 'ara2013_plant054', 'tobacco_plant005', 'ara2012_plant002', 'ara2012_plant052', 'ara2013_plant165', 'ara2013_plant142'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['ara2012_plant057', 'ara2012_plant097', 'ara2012_plant117', 'ara2013_plant147', 'ara2013_plant130', 'ara2013_plant061', 'tobacco_plant025', 'ara2012_plant024'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['tobacco_plant006', 'ara2013_plant102', 'tobacco_plant010', 'ara2013_plant125', 'tobacco_plant019', 'ara2013_plant117', 'ara2013_plant073', 'ara2013_plant040'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['ara2013_plant109', 'tobacco_plant033', 'ara2013_plant094', 'ara2013_plant129', 'ara2012_plant035', 'ara2012_plant062', 'ara2012_plant011', 'ara2013_plant044'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['ara2012_plant068', 'tobacco_plant030', 'ara2012_plant085', 'ara2012_plant050', 'ara2013_plant048', 'ara2013_plant089', 'ara2013_plant100', 'ara2013_plant108'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2012_plant016', 'tobacco_plant045', 'tobacco_plant036', 'ara2013_plant029', 'ara2013_plant033', 'ara2013_plant140', 'tobacco_plant050', 'ara2012_plant001'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['ara2012_plant047', 'tobacco_plant014', 'ara2013_plant149', 'ara2013_plant066', 'ara2012_plant109', 'ara2012_plant116', 'tobacco_plant041', 'ara2012_plant028'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2013_plant068', 'ara2013_plant090', 'ara2013_plant031', 'tobacco_plant062', 'ara2013_plant030', 'tobacco_plant028', 'tobacco_plant017', 'ara2013_plant159'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2013_plant121', 'ara2013_plant064', 'ara2013_plant053', 'ara2012_plant008', 'ara2013_plant032', 'ara2013_plant105', 'ara2013_plant070', 'ara2013_plant123'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['ara2012_plant059', 'ara2013_plant076', 'ara2012_plant036', 'ara2012_plant087', 'ara2013_plant019', 'ara2012_plant031', 'ara2012_plant048', 'ara2013_plant160'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2013_plant020', 'tobacco_plant032', 'ara2013_plant137', 'tobacco_plant051', 'ara2012_plant018', 'ara2013_plant010', 'ara2012_plant039', 'ara2013_plant051'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['ara2013_plant009', 'tobacco_plant029', 'ara2013_plant083', 'ara2013_plant132', 'ara2013_plant008', 'ara2012_plant099', 'ara2013_plant091', 'ara2013_plant052'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['tobacco_plant042', 'ara2013_plant026', 'ara2013_plant106', 'ara2013_plant082', 'ara2012_plant074', 'ara2013_plant115', 'ara2012_plant046', 'ara2013_plant104'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2013_plant116', 'ara2013_plant079', 'ara2012_plant060', 'ara2013_plant016', 'ara2013_plant097', 'ara2013_plant023', 'ara2012_plant102', 'ara2013_plant028'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=32 names=['tobacco_plant058', 'ara2012_plant092', 'ara2012_plant009', 'ara2012_plant078', 'ara2013_plant078', 'tobacco_plant059', 'ara2012_plant030', 'ara2013_plant085'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['ara2012_plant006', 'ara2013_plant027', 'ara2013_plant092', 'ara2012_plant075', 'ara2012_plant003', 'ara2013_plant037', 'tobacco_plant001', 'ara2012_plant045'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=34 names=['ara2012_plant114', 'ara2013_plant164', 'ara2013_plant138', 'tobacco_plant039', 'tobacco_plant035', 'ara2013_plant036', 'ara2013_plant018', 'ara2013_plant022'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2012_plant004', 'ara2013_plant148', 'tobacco_plant037', 'ara2013_plant067', 'tobacco_plant013', 'ara2012_plant071', 'ara2012_plant013', 'tobacco_plant015'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['ara2012_plant089', 'ara2012_plant079', 'tobacco_plant038', 'ara2013_plant161', 'ara2013_plant099', 'ara2013_plant163', 'ara2013_plant057', 'ara2012_plant073'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2012_plant033', 'tobacco_plant053', 'ara2013_plant038', 'ara2013_plant084', 'tobacco_plant004', 'tobacco_plant020', 'ara2013_plant146', 'ara2013_plant055'] — 1048576 elemen.
Epoch  13/100 | Loss: 0.1106 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=0 names=['ara2012_plant111', 'tobacco_plant046', 'ara2013_plant053', 'ara2013_plant025', 'ara2012_plant068', 'ara2013_plant099', 'ara2013_plant104', 'ara2013_plant112'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=1 names=['ara2013_plant102', 'ara2012_plant102', 'ara2013_plant082', 'ara2013_plant134', 'ara2013_plant063', 'ara2012_plant036', 'tobacco_plant034', 'ara2013_plant045'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['ara2013_plant013', 'ara2012_plant017', 'ara2012_plant051', 'ara2013_plant129', 'ara2013_plant110', 'tobacco_plant049', 'ara2013_plant097', 'tobacco_plant052'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['ara2013_plant163', 'ara2012_plant028', 'ara2012_plant073', 'ara2013_plant020', 'ara2013_plant056', 'ara2012_plant093', 'ara2012_plant054', 'tobacco_plant001'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['tobacco_plant055', 'ara2013_plant069', 'tobacco_plant007', 'ara2013_plant075', 'ara2012_plant095', 'ara2012_plant071', 'tobacco_plant012', 'ara2013_plant078'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=8 names=['ara2013_plant145', 'ara2013_plant114', 'tobacco_plant002', 'ara2012_plant094', 'ara2013_plant036', 'tobacco_plant040', 'ara2012_plant030', 'ara2012_plant107'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=9 names=['ara2012_plant060', 'ara2013_plant103', 'ara2013_plant016', 'tobacco_plant062', 'ara2012_plant103', 'ara2013_plant151', 'ara2012_plant006', 'ara2013_plant068'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['tobacco_plant015', 'tobacco_plant036', 'ara2013_plant089', 'ara2013_plant067', 'ara2013_plant061', 'ara2013_plant106', 'ara2013_plant165', 'ara2012_plant008'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['ara2013_plant044', 'ara2013_plant070', 'ara2012_plant104', 'ara2012_plant074', 'ara2013_plant058', 'ara2012_plant029', 'ara2013_plant162', 'ara2013_plant083'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2013_plant153', 'ara2012_plant088', 'ara2013_plant042', 'ara2012_plant022', 'ara2013_plant041', 'tobacco_plant030', 'ara2012_plant020', 'ara2013_plant126'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2012_plant037', 'ara2012_plant001', 'tobacco_plant024', 'ara2013_plant096', 'ara2013_plant023', 'ara2012_plant078', 'ara2013_plant160', 'ara2013_plant157'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['tobacco_plant006', 'ara2013_plant062', 'ara2012_plant032', 'tobacco_plant013', 'ara2013_plant092', 'ara2013_plant143', 'ara2012_plant009', 'tobacco_plant060'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2013_plant094', 'ara2013_plant121', 'ara2013_plant032', 'ara2012_plant108', 'ara2013_plant130', 'ara2013_plant142', 'ara2012_plant039', 'ara2013_plant154'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['ara2012_plant106', 'ara2013_plant006', 'ara2013_plant111', 'ara2012_plant025', 'ara2012_plant080', 'ara2012_plant026', 'tobacco_plant019', 'ara2013_plant113'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2013_plant116', 'tobacco_plant042', 'ara2013_plant158', 'ara2013_plant141', 'tobacco_plant047', 'ara2012_plant057', 'ara2012_plant109', 'ara2013_plant008'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2013_plant054', 'ara2013_plant118', 'tobacco_plant021', 'ara2013_plant136', 'ara2012_plant015', 'ara2013_plant085', 'ara2013_plant019', 'tobacco_plant045'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['ara2012_plant066', 'ara2012_plant004', 'ara2013_plant018', 'ara2013_plant027', 'ara2013_plant148', 'ara2013_plant164', 'tobacco_plant020', 'ara2013_plant072'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['ara2012_plant072', 'ara2012_plant035', 'ara2013_plant135', 'ara2013_plant005', 'ara2012_plant113', 'ara2013_plant149', 'tobacco_plant004', 'ara2012_plant012'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2013_plant125', 'ara2012_plant013', 'ara2012_plant050', 'ara2012_plant016', 'tobacco_plant009', 'ara2012_plant024', 'ara2012_plant099', 'ara2013_plant156'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2012_plant014', 'ara2012_plant117', 'ara2013_plant030', 'ara2013_plant049', 'ara2012_plant097', 'ara2012_plant115', 'tobacco_plant039', 'ara2012_plant082'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2013_plant046', 'ara2012_plant046', 'ara2013_plant048', 'tobacco_plant050', 'ara2012_plant065', 'ara2013_plant009', 'ara2013_plant047', 'ara2012_plant053'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['ara2013_plant029', 'ara2013_plant076', 'ara2013_plant039', 'ara2013_plant012', 'ara2013_plant117', 'ara2013_plant007', 'ara2013_plant033', 'ara2013_plant155'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2013_plant090', 'tobacco_plant018', 'tobacco_plant023', 'ara2013_plant150', 'ara2012_plant069', 'ara2012_plant007', 'ara2013_plant031', 'tobacco_plant028'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['ara2013_plant034', 'ara2013_plant137', 'ara2013_plant138', 'ara2013_plant002', 'ara2013_plant051', 'ara2012_plant085', 'ara2012_plant021', 'ara2012_plant040'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['tobacco_plant032', 'ara2013_plant124', 'ara2013_plant080', 'tobacco_plant014', 'ara2013_plant026', 'ara2013_plant022', 'ara2013_plant101', 'ara2013_plant108'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2013_plant161', 'tobacco_plant058', 'ara2013_plant073', 'ara2012_plant120', 'ara2012_plant092', 'ara2013_plant011', 'ara2012_plant003', 'ara2013_plant079'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=32 names=['tobacco_plant041', 'tobacco_plant017', 'ara2012_plant044', 'ara2012_plant075', 'ara2012_plant055', 'ara2013_plant127', 'ara2012_plant114', 'ara2012_plant056'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=34 names=['ara2012_plant019', 'ara2012_plant100', 'ara2013_plant105', 'ara2013_plant064', 'tobacco_plant038', 'ara2013_plant059', 'ara2013_plant052', 'ara2012_plant047'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2013_plant014', 'ara2012_plant042', 'ara2013_plant146', 'ara2012_plant067', 'ara2013_plant132', 'ara2012_plant031', 'ara2013_plant128', 'ara2013_plant081'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['ara2012_plant083', 'ara2013_plant055', 'ara2012_plant101', 'tobacco_plant033', 'tobacco_plant011', 'tobacco_plant053', 'ara2012_plant058', 'ara2012_plant041'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2012_plant077', 'ara2012_plant090', 'ara2013_plant065', 'ara2013_plant159', 'ara2013_plant140', 'tobacco_plant029', 'ara2013_plant071', 'ara2013_plant086'] — 1048576 elemen.
Epoch  14/100 | Loss: 0.1026 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=0 names=['tobacco_plant027', 'ara2013_plant001', 'ara2012_plant106', 'ara2012_plant004', 'ara2012_plant012', 'ara2012_plant075', 'ara2013_plant012', 'tobacco_plant034'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=1 names=['ara2012_plant040', 'ara2013_plant113', 'ara2012_plant003', 'ara2013_plant018', 'ara2013_plant045', 'tobacco_plant020', 'ara2013_plant162', 'ara2012_plant115'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['ara2013_plant159', 'ara2012_plant062', 'ara2013_plant142', 'ara2013_plant089', 'ara2012_plant043', 'ara2012_plant090', 'ara2013_plant145', 'ara2013_plant014'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=3 names=['tobacco_plant042', 'ara2012_plant034', 'ara2013_plant025', 'tobacco_plant044', 'ara2012_plant046', 'tobacco_plant023', 'ara2012_plant054', 'ara2012_plant108'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['ara2012_plant047', 'ara2013_plant066', 'ara2013_plant116', 'ara2013_plant057', 'tobacco_plant049', 'ara2012_plant080', 'ara2012_plant087', 'ara2013_plant059'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2012_plant118', 'ara2012_plant073', 'tobacco_plant028', 'ara2012_plant016', 'ara2013_plant124', 'ara2013_plant011', 'ara2013_plant052', 'tobacco_plant009'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['ara2013_plant048', 'ara2012_plant088', 'tobacco_plant036', 'ara2012_plant015', 'ara2013_plant063', 'ara2013_plant007', 'tobacco_plant035', 'ara2013_plant075'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2013_plant071', 'ara2013_plant117', 'ara2013_plant105', 'tobacco_plant024', 'tobacco_plant060', 'ara2013_plant055', 'ara2012_plant116', 'ara2013_plant112'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=8 names=['ara2012_plant067', 'ara2012_plant052', 'ara2013_plant061', 'ara2013_plant153', 'tobacco_plant002', 'ara2013_plant099', 'ara2012_plant057', 'ara2013_plant147'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=9 names=['ara2013_plant040', 'tobacco_plant016', 'ara2012_plant029', 'ara2013_plant161', 'ara2013_plant125', 'ara2012_plant053', 'ara2013_plant053', 'ara2013_plant047'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['ara2012_plant109', 'ara2012_plant011', 'ara2012_plant099', 'ara2013_plant032', 'tobacco_plant004', 'tobacco_plant046', 'ara2013_plant049', 'tobacco_plant021'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['ara2012_plant006', 'ara2013_plant010', 'ara2013_plant041', 'ara2012_plant060', 'ara2012_plant120', 'ara2012_plant119', 'ara2013_plant026', 'ara2012_plant111'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['tobacco_plant022', 'tobacco_plant048', 'ara2012_plant093', 'ara2013_plant160', 'ara2013_plant126', 'ara2012_plant002', 'ara2012_plant001', 'ara2013_plant128'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['tobacco_plant051', 'ara2013_plant130', 'ara2013_plant120', 'tobacco_plant059', 'ara2013_plant085', 'ara2013_plant157', 'ara2012_plant095', 'ara2013_plant127'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['ara2012_plant021', 'ara2013_plant148', 'ara2013_plant133', 'ara2012_plant068', 'ara2013_plant121', 'ara2012_plant084', 'ara2012_plant059', 'ara2012_plant048'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2013_plant096', 'ara2012_plant031', 'tobacco_plant050', 'ara2013_plant008', 'ara2013_plant101', 'ara2013_plant111', 'ara2012_plant035', 'ara2013_plant108'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2012_plant078', 'tobacco_plant018', 'ara2012_plant076', 'tobacco_plant053', 'ara2013_plant149', 'ara2013_plant016', 'ara2013_plant129', 'ara2012_plant037'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['tobacco_plant039', 'ara2013_plant082', 'ara2013_plant165', 'tobacco_plant062', 'tobacco_plant006', 'ara2012_plant013', 'ara2012_plant018', 'ara2013_plant114'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['ara2013_plant136', 'ara2013_plant027', 'ara2013_plant065', 'tobacco_plant014', 'ara2013_plant078', 'tobacco_plant032', 'ara2013_plant038', 'ara2012_plant008'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2012_plant069', 'tobacco_plant001', 'ara2012_plant039', 'ara2013_plant080', 'ara2013_plant158', 'tobacco_plant047', 'ara2013_plant132', 'ara2012_plant097'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2012_plant032', 'tobacco_plant008', 'ara2013_plant109', 'tobacco_plant026', 'ara2012_plant009', 'ara2012_plant044', 'ara2012_plant023', 'ara2013_plant146'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['ara2012_plant022', 'ara2013_plant062', 'ara2013_plant056', 'ara2013_plant030', 'tobacco_plant045', 'ara2012_plant042', 'ara2012_plant045', 'ara2013_plant006'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2012_plant019', 'ara2012_plant024', 'ara2012_plant101', 'ara2013_plant094', 'ara2012_plant055', 'ara2013_plant034', 'tobacco_plant040', 'ara2013_plant118'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['ara2013_plant100', 'ara2013_plant144', 'ara2012_plant071', 'ara2013_plant154', 'tobacco_plant030', 'ara2013_plant043', 'ara2012_plant058', 'ara2013_plant046'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2012_plant036', 'ara2013_plant039', 'ara2013_plant035', 'ara2013_plant138', 'ara2012_plant028', 'ara2013_plant115', 'ara2013_plant019', 'ara2012_plant077'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['tobacco_plant013', 'tobacco_plant010', 'ara2012_plant005', 'tobacco_plant038', 'ara2012_plant056', 'ara2012_plant072', 'ara2013_plant021', 'ara2012_plant079'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['tobacco_plant037', 'ara2013_plant068', 'tobacco_plant061', 'ara2013_plant076', 'ara2012_plant113', 'tobacco_plant015', 'ara2013_plant037', 'ara2013_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2013_plant084', 'ara2013_plant156', 'tobacco_plant005', 'tobacco_plant033', 'ara2013_plant044', 'ara2013_plant155', 'ara2013_plant090', 'ara2012_plant033'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=32 names=['ara2013_plant110', 'ara2013_plant083', 'ara2013_plant002', 'ara2012_plant110', 'ara2013_plant070', 'ara2012_plant026', 'ara2013_plant042', 'ara2013_plant163'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['ara2013_plant009', 'ara2012_plant065', 'ara2013_plant022', 'ara2013_plant081', 'ara2012_plant094', 'tobacco_plant012', 'ara2012_plant103', 'ara2012_plant102'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=34 names=['ara2012_plant027', 'ara2013_plant028', 'tobacco_plant058', 'ara2013_plant102', 'ara2013_plant141', 'ara2012_plant081', 'tobacco_plant017', 'ara2012_plant083'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2013_plant067', 'tobacco_plant029', 'tobacco_plant011', 'ara2013_plant064', 'ara2013_plant036', 'ara2013_plant054', 'tobacco_plant025', 'ara2013_plant098'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=36 names=['ara2013_plant073', 'ara2013_plant091', 'ara2012_plant030', 'ara2013_plant086', 'ara2013_plant069', 'ara2013_plant020', 'ara2013_plant072', 'ara2012_plant074'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['ara2013_plant033', 'ara2013_plant164', 'tobacco_plant041', 'ara2013_plant137', 'ara2013_plant104', 'ara2013_plant097', 'ara2013_plant106', 'ara2012_plant107'] — 1048576 elemen.
Epoch  15/100 | Loss: 0.1176 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=0 names=['ara2013_plant148', 'ara2012_plant032', 'ara2012_plant024', 'tobacco_plant036', 'ara2013_plant078', 'ara2013_plant123', 'tobacco_plant011', 'ara2012_plant100'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=1 names=['tobacco_plant035', 'tobacco_plant037', 'ara2012_plant078', 'ara2012_plant113', 'ara2012_plant102', 'ara2013_plant146', 'ara2012_plant090', 'ara2013_plant034'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=3 names=['ara2013_plant102', 'ara2012_plant060', 'ara2013_plant114', 'tobacco_plant061', 'tobacco_plant007', 'ara2013_plant054', 'ara2012_plant084', 'ara2013_plant065'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['tobacco_plant004', 'ara2013_plant033', 'ara2013_plant115', 'ara2013_plant036', 'tobacco_plant049', 'tobacco_plant028', 'ara2013_plant165', 'tobacco_plant026'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2013_plant105', 'ara2013_plant067', 'ara2012_plant027', 'tobacco_plant038', 'ara2013_plant091', 'ara2012_plant085', 'ara2013_plant059', 'ara2012_plant030'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['ara2013_plant160', 'ara2012_plant072', 'ara2013_plant101', 'tobacco_plant062', 'ara2013_plant079', 'ara2012_plant017', 'tobacco_plant030', 'ara2013_plant066'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2013_plant011', 'ara2013_plant007', 'tobacco_plant012', 'ara2012_plant111', 'ara2012_plant120', 'ara2012_plant031', 'ara2012_plant047', 'ara2013_plant009'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=8 names=['ara2013_plant118', 'ara2013_plant121', 'ara2012_plant071', 'tobacco_plant053', 'tobacco_plant059', 'ara2012_plant026', 'ara2013_plant072', 'ara2012_plant004'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['ara2012_plant029', 'ara2012_plant094', 'ara2012_plant002', 'ara2013_plant015', 'ara2012_plant081', 'tobacco_plant046', 'ara2013_plant142', 'ara2013_plant098'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['ara2013_plant052', 'ara2012_plant119', 'ara2012_plant023', 'ara2012_plant025', 'ara2013_plant090', 'ara2013_plant040', 'ara2013_plant016', 'ara2012_plant110'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2012_plant089', 'ara2012_plant044', 'ara2012_plant039', 'ara2012_plant097', 'tobacco_plant009', 'ara2013_plant051', 'ara2013_plant018', 'ara2012_plant063'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2012_plant028', 'ara2013_plant039', 'ara2013_plant133', 'ara2013_plant006', 'ara2013_plant163', 'ara2013_plant153', 'ara2013_plant035', 'ara2013_plant083'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['ara2013_plant089', 'ara2012_plant009', 'ara2012_plant117', 'tobacco_plant047', 'ara2013_plant073', 'ara2012_plant104', 'ara2013_plant068', 'ara2013_plant002'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2013_plant019', 'ara2013_plant012', 'ara2013_plant043', 'ara2012_plant018', 'ara2012_plant033', 'ara2013_plant055', 'ara2013_plant097', 'ara2013_plant161'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['tobacco_plant021', 'ara2012_plant008', 'ara2012_plant062', 'ara2013_plant082', 'ara2013_plant144', 'tobacco_plant017', 'ara2012_plant093', 'ara2012_plant073'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2012_plant016', 'ara2012_plant053', 'ara2012_plant099', 'ara2012_plant014', 'ara2013_plant158', 'ara2012_plant115', 'ara2013_plant064', 'ara2013_plant049'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=20 names=['ara2013_plant084', 'ara2013_plant063', 'ara2012_plant105', 'ara2013_plant071', 'ara2013_plant056', 'ara2013_plant110', 'ara2013_plant081', 'ara2013_plant162'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['tobacco_plant045', 'ara2013_plant129', 'ara2013_plant117', 'tobacco_plant029', 'ara2012_plant080', 'ara2013_plant086', 'ara2013_plant008', 'ara2013_plant164'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['ara2012_plant058', 'ara2012_plant005', 'ara2013_plant125', 'ara2012_plant067', 'ara2012_plant076', 'tobacco_plant023', 'ara2013_plant042', 'ara2013_plant126'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2013_plant096', 'ara2013_plant070', 'ara2013_plant031', 'ara2012_plant092', 'ara2013_plant137', 'ara2013_plant037', 'ara2012_plant040', 'tobacco_plant041'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2013_plant135', 'ara2012_plant056', 'ara2012_plant066', 'ara2012_plant041', 'ara2013_plant151', 'ara2013_plant157', 'ara2012_plant006', 'tobacco_plant060'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['tobacco_plant032', 'ara2013_plant076', 'ara2013_plant058', 'tobacco_plant050', 'ara2012_plant116', 'ara2012_plant045', 'ara2012_plant050', 'ara2013_plant062'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2012_plant069', 'tobacco_plant013', 'ara2013_plant075', 'tobacco_plant014', 'ara2012_plant022', 'ara2012_plant046', 'tobacco_plant019', 'ara2012_plant001'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['ara2013_plant048', 'ara2013_plant053', 'ara2013_plant069', 'ara2012_plant012', 'tobacco_plant051', 'ara2013_plant136', 'ara2013_plant149', 'ara2012_plant036'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2012_plant003', 'ara2013_plant045', 'ara2013_plant116', 'ara2013_plant020', 'ara2013_plant099', 'ara2012_plant021', 'ara2013_plant113', 'ara2012_plant011'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['tobacco_plant055', 'ara2013_plant028', 'tobacco_plant040', 'ara2013_plant023', 'ara2012_plant034', 'ara2013_plant100', 'ara2012_plant054', 'ara2013_plant132'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['tobacco_plant034', 'ara2013_plant108', 'ara2013_plant103', 'ara2012_plant107', 'tobacco_plant039', 'ara2012_plant083', 'tobacco_plant024', 'ara2012_plant109'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['tobacco_plant058', 'ara2013_plant080', 'ara2013_plant001', 'ara2013_plant022', 'tobacco_plant018', 'ara2013_plant104', 'ara2012_plant043', 'ara2013_plant147'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=32 names=['ara2013_plant046', 'ara2012_plant075', 'ara2012_plant108', 'ara2012_plant074', 'ara2013_plant106', 'tobacco_plant002', 'tobacco_plant022', 'ara2013_plant041'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['ara2012_plant095', 'ara2012_plant035', 'ara2012_plant077', 'ara2013_plant094', 'ara2013_plant027', 'ara2012_plant057', 'ara2013_plant010', 'ara2013_plant026'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2013_plant025', 'tobacco_plant010', 'ara2012_plant015', 'ara2013_plant140', 'ara2012_plant106', 'ara2013_plant109', 'ara2013_plant030', 'tobacco_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=36 names=['ara2013_plant138', 'tobacco_plant044', 'ara2012_plant059', 'ara2013_plant127', 'ara2012_plant079', 'ara2013_plant005', 'ara2013_plant014', 'ara2013_plant038'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['ara2012_plant048', 'tobacco_plant042', 'ara2013_plant029', 'tobacco_plant033', 'ara2013_plant150', 'ara2013_plant047', 'ara2013_plant092', 'ara2012_plant114'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2013_plant112', 'ara2012_plant007', 'ara2013_plant134', 'tobacco_plant006', 'tobacco_plant025', 'ara2013_plant145', 'ara2013_plant124', 'ara2012_plant068'] — 1048576 elemen.
Epoch  16/100 | Loss: 0.3687 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=0 names=['ara2012_plant055', 'ara2012_plant081', 'ara2012_plant047', 'ara2013_plant089', 'ara2012_plant083', 'tobacco_plant020', 'ara2013_plant030', 'tobacco_plant033'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=1 names=['tobacco_plant015', 'ara2013_plant109', 'ara2013_plant147', 'ara2013_plant128', 'ara2013_plant165', 'ara2012_plant090', 'ara2013_plant110', 'ara2013_plant046'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['ara2012_plant045', 'ara2013_plant080', 'ara2013_plant047', 'ara2013_plant029', 'ara2012_plant017', 'ara2012_plant094', 'ara2012_plant062', 'ara2012_plant089'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['ara2012_plant025', 'ara2013_plant132', 'ara2013_plant104', 'ara2012_plant069', 'tobacco_plant019', 'ara2012_plant103', 'ara2013_plant091', 'ara2013_plant040'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['tobacco_plant042', 'ara2013_plant135', 'ara2012_plant029', 'ara2013_plant005', 'tobacco_plant018', 'ara2013_plant071', 'ara2013_plant140', 'ara2012_plant057'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['tobacco_plant026', 'ara2012_plant060', 'tobacco_plant052', 'tobacco_plant028', 'ara2012_plant117', 'ara2013_plant058', 'ara2012_plant108', 'ara2013_plant054'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['tobacco_plant060', 'ara2013_plant064', 'ara2013_plant057', 'ara2012_plant012', 'ara2013_plant034', 'ara2013_plant134', 'tobacco_plant041', 'ara2012_plant102'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=8 names=['ara2013_plant083', 'tobacco_plant025', 'ara2013_plant085', 'ara2012_plant030', 'ara2012_plant093', 'ara2013_plant126', 'ara2013_plant035', 'ara2012_plant046'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=9 names=['tobacco_plant061', 'ara2013_plant039', 'ara2012_plant051', 'ara2012_plant059', 'ara2013_plant143', 'tobacco_plant012', 'ara2013_plant001', 'ara2013_plant027'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['ara2013_plant067', 'ara2012_plant040', 'ara2013_plant066', 'ara2012_plant050', 'ara2013_plant044', 'ara2013_plant164', 'tobacco_plant044', 'ara2012_plant088'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['ara2013_plant049', 'ara2013_plant114', 'tobacco_plant053', 'ara2013_plant025', 'ara2013_plant014', 'ara2013_plant055', 'ara2013_plant048', 'tobacco_plant008'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2013_plant006', 'ara2013_plant155', 'ara2013_plant028', 'ara2012_plant043', 'ara2013_plant113', 'ara2013_plant042', 'tobacco_plant009', 'ara2012_plant056'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2012_plant022', 'ara2012_plant042', 'ara2013_plant137', 'ara2012_plant115', 'ara2013_plant120', 'ara2013_plant146', 'ara2012_plant100', 'ara2012_plant044'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['ara2013_plant002', 'ara2013_plant043', 'ara2012_plant053', 'ara2013_plant051', 'tobacco_plant029', 'ara2013_plant084', 'ara2013_plant163', 'ara2013_plant020'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['ara2013_plant010', 'ara2013_plant037', 'ara2012_plant078', 'ara2012_plant037', 'ara2012_plant035', 'ara2013_plant148', 'ara2012_plant052', 'ara2012_plant054'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2013_plant041', 'tobacco_plant051', 'ara2013_plant026', 'tobacco_plant030', 'ara2013_plant136', 'ara2013_plant141', 'ara2012_plant114', 'ara2013_plant023'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['ara2013_plant153', 'ara2013_plant032', 'ara2013_plant031', 'ara2013_plant106', 'tobacco_plant001', 'ara2012_plant027', 'ara2013_plant096', 'ara2013_plant098'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2012_plant024', 'tobacco_plant022', 'ara2013_plant111', 'ara2012_plant015', 'ara2013_plant076', 'ara2013_plant099', 'ara2013_plant151', 'ara2012_plant074'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2013_plant161', 'ara2012_plant104', 'ara2013_plant062', 'ara2013_plant159', 'ara2012_plant023', 'ara2012_plant119', 'ara2012_plant034', 'ara2012_plant106'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=20 names=['ara2012_plant002', 'ara2012_plant113', 'ara2012_plant001', 'tobacco_plant045', 'ara2013_plant125', 'ara2013_plant073', 'tobacco_plant046', 'ara2013_plant139'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['ara2013_plant117', 'ara2013_plant061', 'ara2012_plant107', 'tobacco_plant034', 'tobacco_plant011', 'ara2012_plant099', 'ara2013_plant086', 'tobacco_plant024'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['ara2013_plant138', 'ara2012_plant003', 'tobacco_plant021', 'ara2013_plant112', 'ara2013_plant116', 'tobacco_plant032', 'ara2012_plant039', 'ara2012_plant079'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2013_plant115', 'ara2013_plant022', 'tobacco_plant050', 'ara2012_plant021', 'ara2012_plant080', 'ara2013_plant012', 'ara2013_plant097', 'ara2013_plant065'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2013_plant160', 'tobacco_plant040', 'ara2013_plant130', 'tobacco_plant036', 'ara2012_plant110', 'ara2013_plant072', 'ara2012_plant020', 'ara2013_plant103'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['ara2013_plant144', 'ara2013_plant121', 'ara2012_plant033', 'ara2012_plant032', 'ara2012_plant085', 'ara2013_plant094', 'ara2013_plant100', 'ara2013_plant052'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2012_plant116', 'ara2013_plant145', 'ara2013_plant015', 'ara2013_plant063', 'ara2013_plant053', 'ara2012_plant111', 'ara2012_plant118', 'ara2012_plant095'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['tobacco_plant006', 'ara2013_plant105', 'ara2012_plant067', 'tobacco_plant007', 'tobacco_plant010', 'ara2013_plant019', 'ara2013_plant016', 'ara2013_plant154'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['tobacco_plant048', 'ara2013_plant149', 'ara2012_plant072', 'ara2013_plant118', 'ara2012_plant048', 'ara2012_plant066', 'ara2013_plant157', 'ara2013_plant078'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['ara2013_plant013', 'ara2012_plant082', 'tobacco_plant039', 'ara2013_plant068', 'tobacco_plant058', 'tobacco_plant059', 'ara2012_plant008', 'ara2013_plant124'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2012_plant013', 'ara2013_plant142', 'ara2012_plant006', 'ara2012_plant071', 'ara2013_plant162', 'ara2012_plant101', 'ara2012_plant076', 'ara2012_plant036'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=32 names=['ara2013_plant090', 'ara2012_plant005', 'ara2012_plant004', 'ara2012_plant084', 'tobacco_plant004', 'ara2013_plant011', 'ara2013_plant075', 'tobacco_plant049'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['tobacco_plant013', 'tobacco_plant038', 'ara2013_plant123', 'tobacco_plant014', 'ara2012_plant014', 'ara2013_plant101', 'ara2012_plant007', 'tobacco_plant035'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2012_plant075', 'ara2013_plant009', 'ara2013_plant033', 'ara2013_plant156', 'ara2013_plant127', 'tobacco_plant016', 'tobacco_plant017', 'ara2012_plant097'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=36 names=['ara2012_plant077', 'ara2013_plant008', 'ara2013_plant079', 'ara2013_plant021', 'ara2012_plant016', 'ara2013_plant056', 'ara2013_plant102', 'tobacco_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['tobacco_plant023', 'tobacco_plant037', 'ara2012_plant120', 'ara2012_plant031', 'ara2012_plant058', 'ara2013_plant129', 'ara2012_plant009', 'ara2012_plant019'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2013_plant081', 'ara2013_plant018', 'ara2013_plant092', 'ara2013_plant038', 'ara2012_plant018', 'ara2013_plant082', 'tobacco_plant027', 'ara2013_plant070'] — 1048576 elemen.
Epoch  17/100 | Loss: 0.1029 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=1 names=['ara2013_plant054', 'ara2012_plant017', 'ara2013_plant115', 'ara2013_plant012', 'ara2012_plant035', 'ara2012_plant051', 'ara2012_plant007', 'ara2013_plant155'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['ara2013_plant026', 'ara2012_plant044', 'ara2013_plant161', 'ara2013_plant042', 'ara2013_plant160', 'ara2012_plant079', 'tobacco_plant020', 'ara2013_plant100'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=3 names=['ara2013_plant118', 'ara2013_plant108', 'ara2012_plant018', 'ara2013_plant089', 'ara2012_plant106', 'ara2013_plant007', 'ara2013_plant041', 'ara2013_plant033'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['tobacco_plant027', 'ara2012_plant030', 'ara2013_plant049', 'ara2012_plant116', 'ara2012_plant078', 'ara2012_plant114', 'ara2013_plant067', 'tobacco_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2012_plant063', 'ara2012_plant077', 'ara2013_plant011', 'tobacco_plant021', 'tobacco_plant023', 'ara2013_plant104', 'ara2013_plant091', 'ara2012_plant058'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['tobacco_plant039', 'tobacco_plant025', 'ara2012_plant023', 'ara2013_plant113', 'ara2013_plant137', 'ara2013_plant030', 'ara2012_plant060', 'ara2013_plant110'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2012_plant082', 'ara2013_plant136', 'ara2012_plant033', 'ara2013_plant121', 'ara2013_plant048', 'ara2012_plant036', 'ara2013_plant092', 'ara2012_plant087'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=8 names=['tobacco_plant010', 'tobacco_plant026', 'ara2013_plant037', 'ara2012_plant003', 'tobacco_plant037', 'tobacco_plant051', 'ara2013_plant068', 'ara2012_plant074'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=9 names=['ara2013_plant065', 'ara2013_plant020', 'ara2012_plant069', 'ara2012_plant032', 'ara2012_plant005', 'ara2012_plant019', 'ara2013_plant014', 'ara2013_plant006'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['ara2012_plant057', 'ara2013_plant143', 'ara2013_plant159', 'tobacco_plant038', 'ara2012_plant026', 'tobacco_plant032', 'tobacco_plant035', 'ara2013_plant070'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['ara2013_plant016', 'ara2012_plant015', 'ara2013_plant148', 'ara2012_plant066', 'ara2013_plant123', 'tobacco_plant055', 'ara2013_plant125', 'ara2012_plant108'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2013_plant040', 'ara2013_plant149', 'tobacco_plant028', 'ara2012_plant118', 'ara2013_plant046', 'ara2013_plant145', 'ara2012_plant068', 'ara2012_plant073'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2013_plant134', 'ara2013_plant124', 'tobacco_plant042', 'ara2013_plant051', 'ara2012_plant009', 'ara2013_plant013', 'tobacco_plant016', 'ara2012_plant089'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['tobacco_plant045', 'ara2013_plant062', 'tobacco_plant017', 'ara2013_plant005', 'ara2013_plant151', 'ara2013_plant106', 'ara2013_plant141', 'tobacco_plant022'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['ara2012_plant016', 'tobacco_plant014', 'ara2012_plant048', 'ara2012_plant001', 'ara2013_plant132', 'ara2013_plant019', 'tobacco_plant050', 'ara2013_plant058'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2013_plant023', 'ara2012_plant034', 'ara2013_plant018', 'tobacco_plant060', 'ara2013_plant162', 'ara2013_plant144', 'tobacco_plant002', 'ara2013_plant031'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['ara2013_plant102', 'ara2012_plant008', 'ara2012_plant053', 'ara2013_plant117', 'ara2012_plant120', 'tobacco_plant009', 'ara2012_plant055', 'ara2012_plant101'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['tobacco_plant048', 'ara2012_plant041', 'ara2013_plant036', 'ara2012_plant027', 'ara2013_plant039', 'ara2013_plant079', 'ara2012_plant004', 'ara2013_plant150'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2013_plant043', 'ara2012_plant117', 'ara2013_plant111', 'ara2012_plant113', 'ara2013_plant063', 'ara2013_plant086', 'ara2013_plant032', 'ara2013_plant099'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=20 names=['tobacco_plant029', 'tobacco_plant044', 'ara2013_plant055', 'ara2013_plant120', 'ara2013_plant021', 'ara2012_plant020', 'ara2013_plant057', 'ara2013_plant073'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['ara2013_plant130', 'ara2012_plant002', 'ara2012_plant021', 'tobacco_plant019', 'ara2012_plant028', 'tobacco_plant058', 'ara2012_plant103', 'tobacco_plant061'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['tobacco_plant011', 'tobacco_plant015', 'ara2013_plant114', 'ara2013_plant028', 'tobacco_plant059', 'tobacco_plant024', 'ara2013_plant157', 'ara2013_plant140'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2013_plant069', 'ara2012_plant052', 'ara2013_plant128', 'tobacco_plant012', 'ara2013_plant109', 'tobacco_plant001', 'ara2013_plant116', 'ara2013_plant126'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2013_plant146', 'ara2012_plant039', 'ara2013_plant098', 'ara2012_plant013', 'tobacco_plant004', 'ara2012_plant093', 'tobacco_plant033', 'ara2013_plant027'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['ara2013_plant038', 'ara2013_plant153', 'ara2013_plant059', 'ara2012_plant088', 'ara2013_plant129', 'tobacco_plant007', 'ara2013_plant025', 'ara2012_plant092'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2012_plant095', 'ara2013_plant165', 'ara2013_plant084', 'tobacco_plant062', 'ara2013_plant101', 'ara2013_plant044', 'ara2012_plant054', 'ara2013_plant071'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['ara2012_plant043', 'ara2013_plant029', 'ara2013_plant085', 'ara2013_plant158', 'ara2012_plant022', 'ara2012_plant084', 'ara2012_plant065', 'tobacco_plant046'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2013_plant138', 'tobacco_plant047', 'ara2013_plant064', 'ara2013_plant075', 'ara2012_plant080', 'ara2013_plant112', 'ara2012_plant109', 'ara2012_plant083'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['ara2012_plant045', 'ara2013_plant009', 'ara2013_plant056', 'ara2012_plant094', 'ara2012_plant115', 'ara2012_plant006', 'tobacco_plant036', 'ara2013_plant010'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['ara2013_plant022', 'ara2013_plant094', 'ara2012_plant081', 'ara2012_plant107', 'tobacco_plant041', 'ara2013_plant139', 'ara2013_plant097', 'ara2013_plant061'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=32 names=['ara2013_plant156', 'ara2012_plant097', 'ara2012_plant029', 'ara2013_plant078', 'tobacco_plant040', 'ara2013_plant147', 'ara2013_plant015', 'ara2012_plant012'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['ara2013_plant083', 'ara2012_plant011', 'ara2013_plant045', 'ara2012_plant067', 'ara2013_plant096', 'ara2012_plant090', 'ara2012_plant047', 'ara2012_plant076'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=34 names=['ara2012_plant105', 'tobacco_plant018', 'ara2012_plant102', 'ara2013_plant066', 'ara2013_plant163', 'ara2013_plant072', 'ara2013_plant135', 'ara2012_plant075'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2013_plant103', 'tobacco_plant034', 'ara2012_plant031', 'ara2012_plant110', 'ara2013_plant002', 'ara2013_plant053', 'ara2013_plant035', 'ara2013_plant034'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=36 names=['ara2012_plant042', 'ara2013_plant081', 'ara2012_plant085', 'ara2012_plant072', 'ara2012_plant046', 'ara2012_plant050', 'ara2013_plant052', 'ara2013_plant127'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['ara2013_plant142', 'ara2013_plant082', 'tobacco_plant008', 'ara2013_plant105', 'ara2012_plant099', 'ara2012_plant104', 'ara2013_plant154', 'ara2012_plant071'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2013_plant080', 'ara2012_plant059', 'tobacco_plant013', 'ara2012_plant037', 'ara2012_plant025', 'ara2012_plant100', 'tobacco_plant030', 'ara2012_plant056'] — 1048576 elemen.
Epoch  18/100 | Loss: 0.1572 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
[WARN (C)] pred NON-FINITE at bi=1 names=['tobacco_plant023', 'ara2012_plant062', 'tobacco_plant015', 'ara2012_plant084', 'ara2012_plant016', 'tobacco_plant045', 'ara2012_plant015', 'ara2012_plant093'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=2 names=['tobacco_plant049', 'ara2012_plant033', 'ara2013_plant025', 'ara2012_plant080', 'ara2013_plant155', 'ara2013_plant045', 'ara2012_plant058', 'ara2012_plant097'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=3 names=['ara2013_plant062', 'tobacco_plant052', 'tobacco_plant033', 'tobacco_plant018', 'ara2013_plant099', 'ara2012_plant094', 'tobacco_plant035', 'ara2012_plant076'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=4 names=['ara2012_plant114', 'ara2013_plant096', 'ara2013_plant063', 'ara2013_plant040', 'ara2013_plant072', 'ara2013_plant120', 'ara2013_plant160', 'tobacco_plant053'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=5 names=['ara2013_plant156', 'ara2012_plant077', 'ara2012_plant100', 'ara2012_plant095', 'tobacco_plant042', 'tobacco_plant024', 'ara2012_plant005', 'ara2013_plant130'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=6 names=['ara2012_plant045', 'ara2013_plant114', 'ara2013_plant015', 'ara2013_plant151', 'tobacco_plant007', 'ara2012_plant115', 'ara2013_plant071', 'ara2012_plant085'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=7 names=['ara2012_plant006', 'ara2012_plant113', 'ara2013_plant132', 'ara2013_plant157', 'tobacco_plant037', 'ara2012_plant036', 'tobacco_plant061', 'ara2012_plant073'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=9 names=['ara2012_plant108', 'ara2012_plant057', 'ara2013_plant112', 'ara2013_plant090', 'ara2013_plant106', 'tobacco_plant028', 'ara2013_plant124', 'ara2013_plant153'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=10 names=['ara2013_plant027', 'ara2012_plant087', 'ara2013_plant055', 'ara2013_plant123', 'ara2012_plant034', 'ara2012_plant026', 'ara2013_plant161', 'ara2013_plant163'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=11 names=['ara2013_plant038', 'ara2013_plant010', 'ara2012_plant027', 'ara2013_plant125', 'ara2013_plant084', 'ara2013_plant100', 'ara2012_plant013', 'ara2012_plant035'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=12 names=['ara2013_plant140', 'ara2013_plant104', 'ara2012_plant055', 'ara2013_plant023', 'tobacco_plant041', 'ara2013_plant051', 'ara2013_plant043', 'ara2013_plant121'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=13 names=['ara2013_plant067', 'ara2013_plant128', 'ara2013_plant143', 'ara2013_plant011', 'tobacco_plant008', 'ara2013_plant034', 'ara2013_plant006', 'ara2013_plant089'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=14 names=['ara2013_plant147', 'ara2013_plant022', 'ara2012_plant029', 'ara2012_plant107', 'ara2013_plant019', 'ara2013_plant032', 'tobacco_plant050', 'ara2012_plant118'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=15 names=['ara2013_plant146', 'ara2013_plant026', 'ara2013_plant028', 'ara2012_plant051', 'tobacco_plant055', 'ara2012_plant037', 'ara2013_plant076', 'ara2013_plant092'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=16 names=['ara2013_plant138', 'tobacco_plant002', 'ara2012_plant056', 'ara2012_plant018', 'ara2013_plant086', 'tobacco_plant021', 'ara2013_plant008', 'tobacco_plant005'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=17 names=['ara2013_plant059', 'ara2012_plant014', 'ara2012_plant052', 'tobacco_plant026', 'ara2012_plant012', 'ara2013_plant103', 'ara2013_plant068', 'ara2013_plant036'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=18 names=['ara2012_plant043', 'tobacco_plant004', 'tobacco_plant051', 'ara2012_plant050', 'ara2012_plant041', 'ara2012_plant002', 'ara2013_plant056', 'tobacco_plant014'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=19 names=['ara2013_plant007', 'ara2012_plant083', 'ara2012_plant021', 'ara2013_plant148', 'ara2013_plant058', 'ara2013_plant117', 'ara2013_plant054', 'ara2013_plant105'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=20 names=['ara2013_plant001', 'tobacco_plant034', 'ara2013_plant141', 'ara2012_plant031', 'ara2013_plant046', 'tobacco_plant059', 'ara2013_plant162', 'ara2012_plant101'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=21 names=['ara2013_plant154', 'ara2013_plant053', 'ara2013_plant082', 'tobacco_plant048', 'tobacco_plant038', 'ara2013_plant110', 'ara2012_plant001', 'ara2013_plant109'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=22 names=['tobacco_plant012', 'ara2013_plant052', 'ara2012_plant074', 'ara2012_plant004', 'ara2012_plant099', 'ara2013_plant057', 'ara2012_plant120', 'ara2012_plant003'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=23 names=['ara2013_plant126', 'ara2013_plant080', 'tobacco_plant032', 'ara2013_plant085', 'tobacco_plant020', 'tobacco_plant027', 'tobacco_plant047', 'ara2013_plant165'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=24 names=['ara2013_plant070', 'ara2013_plant061', 'ara2013_plant137', 'ara2013_plant091', 'ara2012_plant072', 'ara2012_plant047', 'ara2013_plant033', 'ara2012_plant092'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=25 names=['ara2013_plant021', 'ara2012_plant106', 'ara2013_plant083', 'ara2013_plant031', 'ara2012_plant110', 'tobacco_plant046', 'ara2012_plant079', 'ara2013_plant069'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=26 names=['ara2013_plant108', 'ara2012_plant059', 'ara2012_plant063', 'ara2012_plant065', 'ara2012_plant039', 'ara2013_plant029', 'ara2013_plant113', 'ara2012_plant017'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=27 names=['tobacco_plant016', 'ara2012_plant105', 'ara2013_plant065', 'ara2012_plant071', 'ara2013_plant020', 'tobacco_plant025', 'ara2012_plant089', 'ara2013_plant044'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=28 names=['ara2012_plant069', 'ara2012_plant024', 'ara2012_plant103', 'ara2013_plant158', 'tobacco_plant001', 'ara2012_plant081', 'ara2013_plant127', 'ara2012_plant109'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=29 names=['ara2012_plant116', 'ara2012_plant044', 'ara2013_plant102', 'ara2013_plant115', 'ara2012_plant042', 'ara2013_plant039', 'ara2013_plant164', 'tobacco_plant019'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=30 names=['ara2012_plant068', 'ara2012_plant020', 'tobacco_plant044', 'ara2013_plant118', 'ara2012_plant019', 'ara2013_plant116', 'ara2012_plant032', 'ara2012_plant111'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=31 names=['ara2012_plant060', 'ara2012_plant046', 'tobacco_plant013', 'ara2013_plant009', 'ara2013_plant159', 'tobacco_plant058', 'ara2012_plant082', 'ara2012_plant053'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=32 names=['tobacco_plant040', 'ara2013_plant049', 'ara2013_plant144', 'ara2013_plant012', 'ara2013_plant145', 'ara2012_plant011', 'ara2013_plant066', 'ara2013_plant150'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=33 names=['tobacco_plant022', 'ara2013_plant142', 'ara2012_plant007', 'ara2012_plant119', 'tobacco_plant039', 'ara2013_plant129', 'tobacco_plant017', 'ara2012_plant054'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=35 names=['ara2013_plant048', 'ara2012_plant078', 'ara2013_plant094', 'ara2013_plant042', 'ara2012_plant023', 'tobacco_plant029', 'tobacco_plant006', 'ara2013_plant018'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=36 names=['ara2012_plant104', 'ara2013_plant079', 'ara2012_plant075', 'ara2012_plant028', 'ara2013_plant139', 'ara2013_plant041', 'ara2013_plant047', 'ara2013_plant135'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=37 names=['ara2013_plant013', 'ara2013_plant035', 'ara2012_plant022', 'tobacco_plant009', 'ara2012_plant102', 'tobacco_plant060', 'ara2012_plant066', 'ara2012_plant067'] — 1048576 elemen.
[WARN (C)] pred NON-FINITE at bi=38 names=['ara2013_plant136', 'ara2013_plant002', 'ara2013_plant073', 'ara2012_plant090', 'ara2013_plant081', 'tobacco_plant036', 'tobacco_plant011', 'ara2013_plant016'] — 1048576 elemen.
Epoch  19/100 | Loss: 0.1010 | Val Loss: 0.0000 | IoU: 0.0000 | Dice: 0.0000
Early stopping at epoch 19
  ✓ Saved: best_weighted_bce_dice.pth (124.2 MB)
  ✓ Saved: config_weighted_bce_dice.yaml
  ✓ Saved: metrics_weighted_bce_dice.json

============================================================
GENERATING COMPARISON ARTIFACTS
============================================================
  ✓ Saved comparison CSV: results/comparison/metrics_comparison.csv
  ✓ Saved comparison MD: results/comparison/comparison_summary.md
  ✓ Saved comparison plot: results/comparison/loss_comparison.png
  ✓ Saved: loss_curves.png
  ✓ Saved: dice_curves.png
  ✓ Saved: iou_curves.png

============================================================
COMPARISON COMPLETE
============================================================
Results in: results
Comparison in: results/comparison