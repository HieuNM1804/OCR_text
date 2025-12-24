# OCR with CRNN (EfficientNet-B5 + BiLSTM + CTC)

This repository implements an Optical Character Recognition (OCR) system using a CRNN architecture. It is designed to recognize text in images using a deep learning approach.

##  Features

*   **Backbone**: EfficientNet-B5 (pretrained on ImageNet) for powerful feature extraction.
*   **Sequence Modeling**: Bidirectional LSTM to capture sequence context.
*   **Decoding**: CTC (Connectionist Temporal Classification) with Beam Search for accurate text decoding.
*   **Data Augmentation**: Includes rotation, color jitter, blur, etc., for robust training.

##  Project Structure

```
.
├── config.py           # Configuration settings (paths, hyperparameters)
├── dataset.py          # Dataset loading and augmentation logic
├── model.py            # CRNN model architecture definition
├── train.py            # Training loop with mixed precision (AMP)
├── predict.py          # Batch prediction script
├── infer_random.py     # Visualization script for random test images
├── utils.py            # Utility functions (CTC Beam Search)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HieuNM1804/OCR_text.git
    cd OCR_text
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

##  Dataset Preparation

The project expects the dataset to be organized as follows:

*   **Images**: Placed in a folder (e.g., `dataset/dataset/train/images`).
*   **Labels**: A JSON file mapping filenames to text labels.

**Example `train.json`:**
```json
{
    "image_001.jpg": "Hello World",
    "image_002.jpg": "OCR Testing"
}
```

Update `config.py` to point to your dataset paths:
```python
CONFIG = {
    'train_json': 'dataset/dataset/train.json',
    'train_img_dir': 'dataset/dataset/train/images',
    'valid_json': 'dataset/dataset/valid.json',
    'valid_img_dir': 'dataset/dataset/valid/images',
    # ...
}
```

##  Usage

### Training
To start training the model:
```bash
python train.py
```
The script uses mixed precision training and saves the best model to `output/best_model.pth`.

### Inference (Visualization)
To run inference on random images from the test set and see the results:
```bash
python infer_random.py
```
This generates `random_inference_results.png`.
![Inference Results](random_inference_results.png)

### Batch Prediction
To predict text for a whole folder of images:
```bash
python predict.py
```
Results will be saved to a JSON file.

##  Requirements

*   Python 3.8+
*   PyTorch
*   Torchvision
*   Pillow
*   Matplotlib
*   tqdm
*   editdistance

