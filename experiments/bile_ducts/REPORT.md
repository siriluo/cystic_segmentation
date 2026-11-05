# Bile-duct focused training

**Date:** 1 September 2026  
**Job:** 3064959 (GH200)  
**Checkpoint:** `experiments/bile_ducts/deeplabv3_best.pth`  
**Split:** same patient holdout as `experiments/train_test` (seed 42)

4-class remap: background, common bile duct, cystic duct, other (GB / liver / instrument / artery / Rouvière collapsed). Loss = weighted CE + CBD/CD Dice + CBD↔CD swap penalty. Train crops around the ducts 70% of the time. Early-stop on val CBD+CD Dice (best epoch 42, 0.332).

## Held-out test (48 frames, 9 patients)

| Class | IoU | Dice | vs 8-class Dice |
| --- | ---: | ---: | ---: |
| Common bile duct | 0.130 | **0.195** | 0.380 (worse) |
| Cystic duct | 0.194 | **0.299** | 0.332 (slightly worse) |
| other | 0.623 | 0.759 | — |
| background | 0.575 | 0.719 | — |

Pixel swap (not the main failure): 9.9% of CBD pixels predicted as CD; 7.3% of CD pixels predicted as CBD. CBD recall is only 0.29 — most CBD pixels go to background or `other`.

Val CBD Dice 0.28 / CD Dice 0.39 is also below the 8-class test numbers, so this is not just a bad test split.

Ground-truth two-column previews (original image | annotated mask) are in `gt_previews/` (`index.html` to browse). Green = CBD, red = cystic duct.

The annotations themselves are a large part of the remaining error: CBD and CD are small adjacent polygons on a continuous ICG-green structure, with no visual boundary at the junction. Several held-out cases are missing one of the two ducts on the sampled frame (Case26: no CD; CaseX10: no CBD; CaseX13: neither on that frame).

