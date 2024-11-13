# Update and upgrade the system packages to ensure everything is up to date
sudo apt update && sudo apt upgrade -y

# Clean up unnecessary packages to free up space
sudo apt autoremove

# Download the CUDA 12.6 installer from NVIDIA's website
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run

# Run the CUDA installer script. This installs CUDA on your system
# During the installation process, you may be prompted to accept the license agreement and choose options (e.g., driver installation)
sudo sh cuda_12.6.0_560.28.03_linux.run

# Verify the CUDA installation by checking the version of the CUDA compiler (nvcc)
nvcc --version

# Check if the NVIDIA GPU drivers are properly installed and the GPU is recognized by the system
nvidia-smi
