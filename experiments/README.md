# Experiments

Training writes checkpoints, loss curves, and validation overlays here.

- `deeplabv3_best.pth` — best validation-loss weights (gitignored)
- `train_val_log.txt` — epoch-wise train/val loss
- `loss_curve.png`, `lr_curve.png` — training plots
- `val_result_*.png` — qualitative overlays

Run:

```bash
python scripts/train.py --data-root ./data --experiment-dir ./experiments
```
