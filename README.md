
# Network_IDS

This repository contains the implementation of a Network Intrusion Detection System (IDS) using various machine learning models, including LightGBM and neural networks. The system is built on the UNSW-NB15 dataset, which is an emulated dataset simulating real-world network traffic. The primary goal of this project is to identify and classify different types of network attacks.

## Dataset: UNSW-NB15

The UNSW-NB15 dataset is an emulated dataset that includes approximately 2,500,000 records, each with 49 features. These features are divided into the following groups:

- **Flow features**: Information about IP and port of the source and destination.
- **Basic features**: Packet status, size, time in the system, etc.
- **Content features**: Statistical information about the packet content.
- **Time features**: Statistical information about the packet timing.
- **Aggregated features**: Aggregated values derived from the above features.
- **Labelled features**: Indicates whether the packet is part of an attack and the type of attack, if any.

The dataset includes 9 types of attacks:
- Fuzzers
- Analysis
- Backdoors
- DoS
- Exploits
- Generic
- Reconnaissance
- Shellcode
- Worms

## Pipeline

1. **Load and process data**:
   - Calculate and generate additional features from existing ones.
   - Handle numerical features by capping extreme values and applying log transformation.
   - Reduce the number of unique values for categorical features and apply OneHotEncoder.
   - Normalize data using StandardScaler().
   - Split data into training and validation sets with a ratio of 8:2.

2. **Train and evaluate models**:
   - Train and evaluate multiple models to select the best performing one.

### Evaluation Results

| Model     | Accuracy | F1    | Recall | Precision | Training Time |
|-----------|----------|-------|--------|-----------|---------------|
| MLP       | 0.8252   | 0.8311| 0.8252 | 0.8775    | 106s          |
| SVM       | 0.7417   | 0.7673| 0.6256 | 0.9919    | 692.09s       |
| LightGBM  | 0.9294   | 0.9504| 0.9945 | 0.9101    | 0.98s         |

#### Model Observations

- **MLP**: Shows overfitting, with high validation accuracy but lower test accuracy. Training is relatively fast.
- **SVM**: Shows overfitting with high precision but low scores in other metrics. Training time is significantly longer.
- **LightGBM**: Provides high accuracy and fast training. It is lightweight and effective for classification tasks with numerical and categorical inputs.

### LightGBM Architecture

LightGBM is an ensemble learning method, specifically a gradient boosting method. It builds a strong learner by sequentially adding weak learners following the gradient descent. Key features include:

- **Gradient-based One-Side Sampling (GOSS)**: Retains samples with large gradients during training, optimizing memory usage and reducing training time.
- **Histogram-based Algorithms**: Uses histograms to build decision trees, reducing computational cost.
- **Leaf-wise Tree Growth**: Grows trees leaf-wise, selecting leaves with the largest errors to split, reducing model error faster than level-wise growth.
- **Exclusive Feature Bundling (EFB)**: Groups mutually exclusive features to reduce the number of features, saving memory and speeding up training.

### Key Advantages of LightGBM

- Fast and accurate
- Memory efficient
- Supports parallel and distributed processing
- Handles large datasets effectively

## Monitoring API

The monitoring API captures network packets over a specified period using Scapy and saves them as .pcap files. The .pcap files are then parsed to extract features similar to those in the UNSW-NB15 dataset. These features are used for prediction and results are returned to the client. The process repeats continuously, capturing and processing new packets at regular intervals.

## Usage

### Clone the Repository

```sh
git clone https://github.com/phanhoang1803/Network_IDS
cd Network_IDS/DetectorModels/src
```

### Train Neural Network

To train a neural network, the following arguments are required. For more details and additional arguments, refer to Training parameters (NN-specific) in `utils/utils.py`.

```sh
python train.py \
--data_dir "path/to/data/dir"
```

### Train LightGBM

To train LightGBM, the following arguments are required. For more details and additional arguments, refer to LightGBM parameters in `utils/utils.py`.

```sh
python train_lgbm.py \
--data_dir "path/to/data/dir"
```

### Run the Application

```sh
cd Network_IDS/app
python run.py
```

### Demo

Watch the demo: [Demo](https://youtu.be/_yWFuqgQrCk?si=t6Ozmd028sPnqOvC)

## Conclusion

This project demonstrates the implementation of a Network IDS using the UNSW-NB15 dataset and various machine learning models, with LightGBM being the chosen model due to its superior performance and efficiency. The monitoring API enables real-time packet capture and analysis, making the system practical for real-world applications.
