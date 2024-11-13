# 🚀 CUDA Toolkit Installation Guide

<div align="center">
  <img src="https://img.shields.io/badge/CUDA-12.6-brightgreen?style=for-the-badge" alt="CUDA Version">
  <img src="https://img.shields.io/badge/Platform-Linux%20|%20Windows-blue?style=for-the-badge" alt="Platform">
  <img src="https://img.shields.io/badge/License-MIT-yellowgreen?style=for-the-badge" alt="License">
</div>

<p align="center">
  <a href="#prerequisites">Prerequisites</a> •  <a href="#installation">Installation</a> •
  <a href="#verification">Verification</a> •
  <a href="#troubleshooting">Troubleshooting</a> •
  <a href="#advanced-configuration">Advanced</a>
</p>

---

## 🎯 Prerequisites

<details><summary>Hardware Requirements</summary>

| Component     | Minimum          | Recommended         |
|---------------|------------------|---------------------|
| GPU           | NVIDIA Kepler+   | NVIDIA Ampere/Hopper |
| RAM           | 4GB              | 16GB+               |
| Disk Space    | 2.5GB            | 10GB                |
| CPU           | x86_64           | Multi-core x86_64   |

</details>

<details>
<summary>Software Requirements</summary>

```mermaid
graph LR
    A[Operating System] --> B[Linux/Windows]
    B --> C[Compatible Driver]
    C --> D[Development Tools]
```

- 🖥️ **Operating System**
  - Linux: Kernel 3.10+
  - Windows: 10/11
- 🔧 **Development Tools**
  - GCC 7+ (Linux)  
  - MSVC 2019+ (Windows)
- 🎮 **NVIDIA Driver**: 525.0.0+

</details>

---

## 📦 Installation

### Linux Installation

<details>
<summary>Quick Install (Recommended)</summary>

```bash
# Download CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_535.54.03_linux.run

# Install
sudo sh cuda_12.6.0_535.54.03_linux.run
```
</details>

<details>
<summary>Package Manager Install</summary>

```bash
# For Ubuntu/Debian
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt install cuda-toolkit-12-6
```
</details>

### Windows Installation

<details>
<summary>Network Installer</summary>

1. Download the [CUDA Network Installer](https://developer.nvidia.com/cuda-downloads)
2. Run the installer
3. Follow the wizard 🧙‍♂️
</details>

---

## 🔧 Environment Setup

Add these to your shell configuration file (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
# CUDA Toolkit Path
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

<details><summary>📝 Optional Environment Variables</summary>

```bash
# Optional: CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Optional: Default device
export CUDA_VISIBLE_DEVICES=0

# Optional: CUDA cache path
export CUDA_CACHE_PATH="$HOME/.cuda-cache"
```
</details>

---

## ✅ Verification

```bash
# Check CUDA compiler
nvcc --version

# Check NVIDIA driver
nvidia-smi

# Run sample test
cd $CUDA_HOME/samples/1_Utilities/deviceQuery
make
./deviceQuery
```

Expected Output:
```
📊 CUDA Device Query Results:  CUDA Version: 12.6
  Driver Version: 535.54.03  CUDA Capability: x.y  ...
```

---

## 🔍 Troubleshooting

<details><summary>Common Issues</summary>

| Issue                        | Solution                  |
|------------------------------|---------------------------|
| `nvcc: command not found`     | Check PATH variable       |
| Driver version mismatch      | Update NVIDIA driver      |
| Installation fails            | Check system requirements |
| CUDA not found                | Verify environment variables |

</details>

<details><summary>Diagnostic Commands</summary>

```bash
# Check CUDA installation
ls -l /usr/local/cuda

# Check driver status
systemctl status nvidia-driver

# Check GPU detection
lspci | grep -i nvidia
```
</details>

---

## 🚀 Advanced Configuration

<details>
<summary>Multi-GPU Setup</summary>

```bash
# List all GPUs
nvidia-smi -L

# Set specific GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Check GPU utilization
nvidia-smi -l 1
```
</details>

<details>
<summary>Performance Optimization</summary>

- Enable persistence mode

```bash
sudo nvidia-smi -pm 1
```

- Set GPU clock speeds

```bash
sudo nvidia-smi -ac 5001,1590
```
</details>

---

## 📚 Additional Resources

- [📖 CUDA Documentation](https://docs.nvidia.com/cuda/)
- [🎓 CUDA Training](https://developer.nvidia.com/cuda-training)
- [💻 Sample Projects](https://github.com/NVIDIA/cuda-samples)
- [🗣️ Developer Forums](https://forums.developer.nvidia.com/c/gpu-programming-and-driver-model/cuda/)

---

<div align="center">

### 🌟 Support & Community
[![Forum](https://img.shields.io/badge/-Forum-brightgreen?style=for-the-badge)](https://forums.developer.nvidia.com/)
[![Issues](https://img.shields.io/badge/-Issues-orange?style=for-the-badge)](https://developer.nvidia.com/cuda-toolkit-bugfix-updates)
[![Documentation](https://img.shields.io/badge/-Documentation-blue?style=for-the-badge)](https://docs.nvidia.com/cuda/)

</div>

---
