# Training data (not committed)

Place case folders here. Videos, frames, and VIA annotations are gitignored because they may contain identifiable surgical footage and are too large for GitHub.

## Layout

```
data/
  processed/
    Case01/
      after_cut_no_icg/
        image.jpg
        annotation.json
    Case43/
      ...
  raw_import/            # unpacked zip (images + VIA json/txt only)
```

Rebuild from the uploaded zip:

```bash
python scripts/organize_dataset.py \
  --zip "AI project case pictures.zip" \
  --out data/processed
```

Annotation overlays, videos, and office files are skipped. Duplicate frames are dropped by image hash. Patient IDs come from the deepest `CaseNN` / `CaseXNN` folder or filename.

Each `.txt` file is a VIA JSON object keyed by the image filename. Regions use `shape_attributes` (polygon `all_points_x` / `all_points_y`) and `region_attributes.label`.

## Labels

The loader maps these annotation strings onto training classes:

| Annotation string | Class id | Display name |
| --- | --- | --- |
| `Gallbladder` or `Gallbradder` | 1 | Gallbladder |
| `Commonbileduct` | 2 | Common bile duct |
| `Cysticduct` | 3 | Cystic duct |
| `Instrument` | 4 | Instrument |
| `liveredge` | 5 | Liver edge |
| `Cysticartery` | 6 | Cystic artery |
| `LuviereSulcus` | 7 | Rouvière's sulcus |

Overlapping polygons are resolved in this order: common bile duct, cystic duct, cystic artery, then remaining classes.

See `assets/annotation.json` for a single-image example used by `notebooks/explore_annotations.ipynb`.
