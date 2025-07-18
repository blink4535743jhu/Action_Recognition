# Video Classification with UCF50 using the LRCN model

This project implements an action classification pipeline using the UCF50 dataset. It leverages a Long-term Recurrent Convolutional Network (LRCN) model that extracts spatial features from individual video frames via a ResNet backbone and learns temporal dynamics through an LSTM. The project includes scripts for preprocessing, training, and testing the model.

This project uses a google colab file to assist in file execution. The README is structured around executing cells in Google Colab. However, the commands from these cells can also be executed in an independent environment.
---

## Table of Contents

- [Dataset Preparation](#dataset-preparation)
- [Environment Setup](#environment-setup)
- [Preprocessing and Frame Extraction](#preprocessing-and-frame-extraction)
- [Training the Model](#training-the-model)
- [Testing and Evaluation](#testing-and-evaluation)
- [Project Structure](#project-structure)
- [Customization and Hyperparameters](#customization-and-hyperparameters)

---

## Dataset Preparation

### Step 0: Download and Unzip Dataset

1. **Download Dataset:**  

   The dataset is downloaded in cell 0b, using the following command:
   !wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF50.rar
   Ensure file is downloaded to the /content/Action_Recognition folder (run cell 0a before 0b).
   The UCF50 dataset can alternatively be downloaded from [(https://www.crcv.ucf.edu/data/UCF50.rar)].
   This dataset contains videos of 51 different human action classes.

3. **Unzip and Organize:**  
   Run cell 0c to unzip the downloaded dataset.
   The expected folder structure should be as follows in the google colab environment:

   - Action_Recognition
        - UCF50
            - Action_Class1
            - Action_Class2
            ... ... ... ...

Each subdirectory represents a different action class.

---

## Environment Setup

1. **Python Version:**  
This project requires Python 3.7 or higher.

2. **Dependencies:**  
Install the required Python packages by running step 0a. in the google colab notebook.

"""
!git clone https://github.com/blink4535743jhu/Action_Recognition
%cd Action_Recognition
%pip install -qr requirements.txt
"""

# Key Libraries

- **PyTorch**
- **torchvision**
- **OpenCV**
- **scikit-learn**
- **tqdm**
- **numpy**
- **Pillow**

## Hardware Requirements

This code was run using the Google Colab A100 notebook.
The repository can be executed using a CUDA-enabled GPU is recommended for training. 
The code automatically detects GPU availability.
 
---

## Preprocessing and Frame Extraction
Before training, the raw video files must be converted into frame sequences. 

Run cell 1 to preprocess video frames. The command is as follows in the google colab environemnt:
!python preprocess.py \
  --input_dir "/content/Action_Recognition/UCF50" \
  --output_dir "/content/Preprocessed_UCF50" \
  --num_frames 16
  
The preprocessing module includes functions for:

### Uniform Frame Sampling

- The `get_frames` function uses OpenCV to sample a fixed number of frames per video.

### Saving Frames to Disk

- The `store_frames` function writes the extracted frames as JPEG images.

The resulting folder structure should mirror the original dataset structure:

   - Action_Recognition
        - Preprocessed_UCF50
            - Action_Class1
                 - v_actionclass1_g01_c01
                 - v_actionclass1_g01_c02
            - Action_Class2
            ... ... ... ...

---

## Training the Model

### Step 1: Run Training

#### Configure Training Parameters and Run the Training Script

The training is managed via a bash script (e.g., `train.sh`) that calls the main training module.
To train the model, run cell 2 in the google colab environment. 
If local, execute the training script from your terminal:

```bash
bash train.sh

**Important:** Update the `--frame_dir` argument in the script to point to the directory where your preprocessed frame data is stored. You can also adjust other parameters (e.g., number of frames per video, batch size, learning rate) to see how they affect the experiment.

If using the google colab environment, the folders will be set up properly to link to /content/Action_Recognition/Preprocessed_UCF50.

## During Training, the Script Will:

- **Load the frame dataset.**
- **Split the dataset** into training, validation, and test sets using stratified sampling.
- **Apply data augmentation** techniques (resizing, random flips, affine transformations).
- **Create custom PyTorch Datasets and DataLoaders.**
- **Initialize the LRCN model** using a specified ResNet backbone.
- **Set up the loss function, optimizer, and learning rate scheduler.**
- **Run the training loop** while tracking loss and accuracy, saving the best model weights.

---

## Testing and Evaluation

### Step 2: Run Testing

- **Configure Testing Parameters:**  

Update the `--ckpt` argument in your testing script (e.g., `test.sh`) to point to the saved best model weights generated during training.
If running in google colab, and executing cells in order, the test.sh script will seamlessly link to the weights directory, as shown below:
/content/Action_Recognition/models/best_model_wts.pt

- **Run the Testing Script:**

The testing is managed via a bash script (e.g., `train.sh`) that calls the main testing module.
To test the model, run cell 3 in the google colab environment. 
If local, execute the testing script from your terminal:
  
```bash
bash test.sh
## Testing Script Overview

The testing script will:

- **Load the dataset splits** (previously saved during training).
- **Create a DataLoader for the test set.**
- **Load the trained model checkpoint.**
- **Evaluate the model** on the test data by computing overall accuracy, generating classification reports, and optionally producing confusion matrices.

---

## Customization and Hyperparameters

You can modify several parameters to experiment with different settings:

### Data Parameters

- `--frame_dir`: Path to your preprocessed frames.
- `--fr_per_vid`: Number of frames to sample per video.

### Model Parameters

- `--model_type`: Choose between `'lrcn'` (default) or other supported models.
- `--cnn_backbone`: Options include `resnet18`, `resnet34`, `resnet50`, `resnet101`, or `resnet152`.
- `--rnn_hidden_size` and `--rnn_n_layers`: Configure the LSTM network.

### Training Parameters

- `--batch_size`, `--learning_rate`, `--n_epochs`, and `--dropout` control the training dynamics.
- `--train_size` and `--test_size` determine dataset splits.

By tweaking these parameters, you can study their impact on model performance and experiment with different network configurations.

---

## Summary of Steps

- **Step 0: Dataset Preparation**  
  Run all cell 0 and cell 1 in google colab, or if local: download, unzip, and organize the UCF50 dataset into subdirectories by action class.

- **Step 1: Run Training**  
  Run cell 2 in google colab, or execute `train.sh` after configuring the `--frame_dir` and other hyperparameters to train the model.

- **Step 2: Run Testing**  
  Run cell 3 in Google Colab, or execute `test.sh` after updating the `--ckpt` argument to point to the best model checkpoint to evaluate the model.

Happy Training!
