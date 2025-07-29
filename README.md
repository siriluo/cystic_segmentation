# AI-Assisted Laparoscopic Cholecystectomy Segmentation

## Project Overview
This project develops an advanced deep learning-based segmentation model for identifying critical anatomical landmarks during laparoscopic cholecystectomy procedures. The primary goal is to reduce the incidence of bile duct injury (BDI) through accurate, real-time identification of anatomical structures.

## Clinical Significance
- Current BDI rate remains at ~0.4% despite surgical advances
- ~60% of injuries result from anatomical misidentification
- AI-driven assistance aims to enhance surgical safety and improve patient outcomes

## Key Anatomical Landmarks
The model focuses on segmenting six critical structures:
1. Gallbladder
2. Cystic Duct
3. Cystic Artery
4. Common Bile Duct
5. Liver Edge
6. Surgical Instruments

## Technical Implementation
The project consists of several key components:

### Data Processing
- `dataloader.ipynb`: Handles data loading and preprocessing
- `dataset.py`: Implements custom dataset class for training
- `chunk_video.py`: Processes surgical video data

### Model Training
- `train.py`: Main training script
- `train.sh`: Training execution script
- `validate.py`: Validation script

### Inference
- `inference.py`: Real-time inference implementation

## Project Structure
```
.
├── dataloader.ipynb      # Data loading and visualization
├── dataset.py           # Dataset implementation
├── train.py            # Training implementation
├── validate.py         # Validation script
├── inference.py        # Inference script
├── requirements.txt    # Project dependencies
└── experiment/         # Training logs and results
```

## Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your data:
   - Place your surgical video data in the `data` directory
   - Ensure proper annotation format as specified in `annotation.json`

3. Train the model:
   ```bash
   bash train.sh
   ```

4. Run inference:
   ```bash
   python inference.py
   ```

## Model Architecture
The project utilizes state-of-the-art deep learning techniques for semantic segmentation of surgical videos. The implementation focuses on real-time performance while maintaining high accuracy in identifying critical anatomical structures.

## Future Work
- Enhancement of real-time performance
- Integration with surgical systems
- Expansion of the anatomical structure dataset
- Clinical validation studies

## References
[1] Literature reference for BDI statistics
[2] Literature reference for AI-assisted surgery

## License
[Specify your license here]

## Contact
[Your contact information]
