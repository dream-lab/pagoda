# Pagoda


This repository contains a suite of tools and scripts for analyzing and estimating the performance of Deep Neural Network (DNN) models on GPU hardware, with a focus on FLOPs, memory accesses, and roofline model analysis.

### Overview
This project provides a comprehensive framework to estimate and measure the computational and memory characteristics of various DNN models. It includes:

- Scripts to convert popular PyTorch models (ResNet, BERT, MobileNet, YOLO, LSTM) to the ONNX format.

- Python scripts for static analysis of ONNX models to estimate theoretical FLOPs and memory accesses.

- Bash scripts for automating NVIDIA CUDA profiling (NCU) to capture real-world performance metrics.

- Jupyter notebooks for post-processing and visualizing the collected data, including roofline plots and batch size experiments.

```
Project Structure
├── onnx_models/                   # Directory for generated ONNX models
├── runtime_data/                  # Directory for storing raw and processed data
│   └── ncu_work/                  # NVIDIA Nsight Compute raw data
│   └── csvs/
│       └── ert_results/           # Post-processed CSV files
├── src/
│   ├── analysis/
│   │   ├── create_onnx_models.ipynb    # Jupyter notebook to export models to ONNX
│   │   ├── ncu-files-analysis.ipynb    # Jupyter notebook for NCU data analysis
│   │   └── roofline_batch_size_expt.ipynb  # Notebook for roofline/batch size experiments
│   ├── scripts/
│   │   ├── exp_script.sh              # Helper script for running experiments
│   │   └── inference_automation.sh    # Main script for automating NCU profiling
│   ├── tools/
│   │   ├── ert_method.py              # Script to process NCU data using the ERT method
│   │   ├── flops_mem_estimates.py     # Script for static FLOPs and memory estimation
│   │   ├── inference_estimates.py     # Core logic for ONNX model analysis
│   │   ├── inference_estimates_lstm.py # Specialized logic for LSTM models
│   │   └── ncu_post_processing.py     # Script to calculate FLOPs/memory from NCU reports
├── LICENSE                        # License file
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

### Getting Started
Prerequisites
- Python 3.8+

- CUDA Toolkit and NVIDIA Nsight Compute (NCU)

- Basic Linux shell environment

- Optional: A compatible NVIDIA GPU to run the profiling scripts.

### Installation
**Clone the repository:**

```
git clone [https://github.com/dream-lab/pagoda.git](https://github.com/dream-lab/pagoda.git)
cd pagoda
```

**Install the required Python packages:**
```
pip install -r requirements.txt
```

### Usage
**Generate ONNX Models**
Use the `create_onnx_models.ipynb` Jupyter notebook to export the supported PyTorch models to the ONNX format. The models will be saved in the `onnx_models/` directory.

**Run Performance Analysis**
The `inference_automation.sh` script automates the process of profiling models using NCU. This script is configured to profile models with different batch sizes and save the results in the `runtime_data/ncu_work/` directory.

**Post-processing and Visualization**
The collected NCU data can be analyzed and visualized using the provided Jupyter notebooks.

`ncu-files-analysis.ipynb`: Provides a detailed breakdown of kernel-level performance.

`roofline_batch_size_expt.ipynb`: Contains scripts to analyze performance across different batch sizes and generate roofline plots.

### License
This project is licensed under the Apache License.