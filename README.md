# Pagoda: An Energy and Time Roofline Study for DNN Workloads on Jetson Edge Accelerators

Edge accelerators such as Nvidia Jetsons are becoming an integral part of the computing continuum, and are often used for DNN inferencing and training. 
Nvidia Jetson edge devices have 2000+ CUDA cores within a 70W power envelope and offer 1000s of power modes to customize CPU, GPU and memory frequencies. 
Their widely varying power–performance trade-offs can be exploited for energy and power-constrained deployments. 
While data-driven methods to predict the power and latency of DNN workloads for edge devices exist, there is a lack of principled study to understand why edge accelerators and their power modes perform the way they do. 
Pagoda presents a time roofline and a novel energy roofline model for the Jetson Orin AGX for diverse power modes, coupled with an analytical model of the compute (FLOP) and memory access (bytes) for DNN inference workloads to analyze them from first principles.
These reveal unique, sometimes counter-intuitive, insights into the power and performance behavior of DNN workloads on edge accelerators. It also we applies these methods to tune the power mode (and hence the roofline) of the edge device to optimize the latency and energy for DNN inference, with up to 15% lower energy and minimal degradation in inference time.

This repository contains a suite of tools and scripts used by Pagoda for analyzing and estimating the performance of Deep Neural Network (DNN) models on GPU hardware, with a focus on FLOPs, memory accesses, and roofline model analysis.

### Overview
This project provides a comprehensive framework to estimate and measure the computational and memory characteristics of various DNN models. It includes:

- Scripts to convert popular PyTorch models (ResNet, BERT, MobileNet, YOLO, LSTM) to the ONNX format.

- **Python scripts for analytically estimating the FLOPs, Memory Accesses, and Arithmetic Intensity of DNN models.**

- Bash scripts for automating NVIDIA CUDA profiling (NCU) to capture real-world performance metrics.

- Jupyter notebooks for post-processing and visualizing the collected data, including roofline plots and batch size experiments.

### Project Structure
```
├── src/
│   ├── analytical_model/
│   │   ├── estimates.py           # Core logic for ONNX model analysis
│   │   ├── estimates_lstm.py      # Specialized logic for LSTM models
│   │   └── model.py               # Script for analytical FLOPs and memory estimation
│   ├── analysis/
│   │   ├── create_onnx_models.ipynb    # Jupyter notebook to export models to ONNX
│   │   ├── ncu-files-analysis.ipynb     # Jupyter notebook for NCU data analysis
│   │   └── roofline_batch_size_expt.ipynb  # Notebook for roofline/batch size experiments
│   ├── scripts/
│   │   ├── exp_script.sh              # Helper script for running experiments
│   │   └── inference_automation.sh    # Main script for automating NCU profiling
│   ├── tools/
│   │   ├── ert_method.py              # Script to process NCU data using the ERT method
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
git clone https://github.com/dream-lab/pagoda.git
cd pagoda
```

**Install the required Python packages:**
```
pip install -r requirements.txt
```

### Usage
**Generate ONNX Models**
Use the `create_onnx_models.ipynb` Jupyter notebook to export the supported PyTorch models to the ONNX format. The models will be saved in the `onnx_models/` directory.

To create .onnx file, load the model, specify the input dimensions (keep the batch size 1) and export it. Refer to example cells in the notebook.

**Estimate FLOPs, Memory Accesses, and Arithmetic Intensity**
The `model.py` script uses the ONNX models to estimate the FLOPs, Memory Accesses, and Arithmetic Intensity for a given DNN model. This script can estimate these metrics for both **training** and **inference** workloads. Make sure to pass the correct workload to get the correct estimates.
```bash
python analytical_model/model.py --model_path <path to .onnx file> --model_name <name_of_the_model> --training <set to True if for training> --batch_size <batch size>
```

**Run Performance Analysis**
The `inference_automation.sh` script automates the process of profiling models using NCU. This script is configured to profile models with different batch sizes and save the results in the `runtime_data/ncu_work/` directory.


**Post-processing and Visualization**
The collected NCU data can be analyzed and visualized using the provided Jupyter notebooks.

`ncu-files-analysis.ipynb`: Provides a detailed breakdown of kernel-level performance.

`roofline_batch_size_expt.ipynb`: Contains scripts to analyze performance across different batch sizes and generate roofline plots.

### License
This project is licensed under the Apache License.

## Attribution
Pagoda: An Energy and Time Roofline Study for DNN Workloads on Edge Accelerators, [Prashanthi S. K.](mailto:prashanthis@iisc.ac.in), Kunal Kumar Sahoo, Amartya Ranjan Saikia, Pranav Gupta, Atharva Vinay Joshi, Priyanshu Pansari and [Yogesh Simmhan](mailto:simmhan@iisc.ac.in), Technical report, Indian Institute of Science, 2005, https://github.com/dream-lab/pagoda/
