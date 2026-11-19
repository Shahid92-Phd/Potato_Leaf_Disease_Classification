# Dataset provenance and leakage check (15 Nov 2026)

* Split as distributed (Putra 2020): Train 900 / Valid 300 / Test 300; 300/100/100 per class; all images 256 x 256.
* Exact duplicates (MD5): 0.
* Perceptual-hash candidates (Hamming <= 6, incl. flips/rotations): 54 pairs, 14 across subsets, 15 with different labels.
  All 14 cross-subset pairs were inspected visually (figures/near_duplicates.png): every pair shows two different
  leaves (different shape, lesions and petiole) photographed in a similar pose on the uniform PlantVillage
  background, i.e. false positives. No cross-subset duplicate was confirmed.
* Healthy class: all 500 images are augmented derivatives (file names Potato_healthy-<source>-_0_<n>.jpg) of
  125 source leaves, exactly 4 per source. Sources 76-150 are in Train (75 leaves), 51-75 in Valid (25),
  26-50 in Test (25). No source leaf occurs in more than one subset -> no leakage, but the healthy test class
  represents 25 independent leaves (100 images).
* Early and late blight: 1,000 original PlantVillage images (unique file names).
* test_dedup.csv therefore contains all 300 test images (no de-duplicated subset is needed).
