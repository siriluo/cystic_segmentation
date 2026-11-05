# 3-class ducts-only training

**Date:** 1 September 2026  
**Job:** 3065146  
**Task:** `ducts-only` — background / common bile duct / cystic duct (GB, liver, instruments, artery, Rouvière all mapped to background)  
**Split:** same as `experiments/train_test` (seed 42)

Same loss (CE + CBD/CD Dice + swap penalty) and duct crops as the 4-class run.

## Held-out test (48 frames, 9 patients)

| Setup | CBD Dice | CD Dice |
| --- | ---: | ---: |
| 8-class landmarks | 0.380 | 0.332 |
| 4-class (CBD / CD / other) | 0.195 | 0.299 |
| **3-class (CBD / CD / background)** | **0.204** | **0.329** |

3-class matches 8-class on cystic duct and is slightly better than 4-class, but CBD stays ~0.20. Background Dice 0.92 is high only because almost every pixel is background.

CBD recall 0.32; most missed CBD pixels go to background. CD recall 0.70, but 12% of CD pixels are predicted as CBD. Without an `other` class, many liver/GB pixels are painted as ducts (false positives).
