# **CUDA Installation Guide**

This guide will walk you through the process of installing **CUDA** on your machine, ensuring you can take full advantage of the GPU for deep learning and computational tasks. The instructions are optimized for both novice and advanced users.

---

## **Prerequisites**

Before proceeding with the installation, ensure the following:

1. **NVIDIA GPU** installed on your system (you can verify this using the `nvidia-smi` command).
2. **NVIDIA drivers** should be installed and configured correctly for your GPU.
3. An updated system with **sudo** privileges for installation steps.

---

## **Step-by-Step Installation**

### **1. System Update**

Open a terminal and update your system packages to ensure you're starting with the latest software:

```bash
sudo apt update && sudo apt upgrade -y && sudo apt autoremove
```

This ensures your system is up to date and cleans up any unnecessary packages.

---

### **2. Download CUDA Toolkit**

Go to the official [CUDA Downloads Page](https://developer.nvidia.com/cuda-downloads) and fill in the following details:

- **Operating System**: Select your OS (Ubuntu, CentOS, etc.).
- **Architecture**: Choose `x86_64` for most modern systems.
- **Distribution**: Choose the correct Linux distribution.
- **Version**: Select the version you want to install (e.g., 12.6.0).
- **Installer Type**: Choose **`runfile`** as the installer type.

Click on the **Download** button to get the installer.

Alternatively, you can directly download the installation file using `wget`:

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
```

This command downloads the CUDA installer package for version 12.6.0.

---

### **3. Install CUDA**

After downloading, navigate to the directory where the `.run` file is saved. Execute the installation command:

```bash
sudo sh cuda_12.6.0_560.28.03_linux.run
```

During installation, you’ll be prompted with some options. You can proceed with the default options, but be mindful of:

- **Driver installation**: You may choose to install or skip installing NVIDIA drivers if they are already installed on your system.
- **Toolkit installation**: Ensure that the CUDA Toolkit is installed.

After the installation is complete, you can verify it by checking the **CUDA version** using the `nvcc` command.

---

### **4. Verify Installation**

To check the installed version of CUDA, run:

```bash
nvcc --version
```

You should see something similar to:

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Sep_15_22:25:13_PDT_2024
Cuda compilation tools, release 12.6, V12.6.0
```

Additionally, run `nvidia-smi` to confirm that your GPU is recognized by CUDA:

```bash
nvidia-smi
```

You should see information about your GPU and CUDA version.

---

### **5. Update Environment Variables (if needed)**

If `nvcc` doesn’t work right away, the issue might be with your environment variables. First, check which shell you’re using:

```bash
echo $SHELL
```

- If it returns `/bin/bash`, modify the **`~/.bashrc`** file.
- If it returns `/bin/zsh`, modify the **`~/.zshrc`** file.

Add the following lines to the appropriate configuration file:

```bash
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
```

Then, apply the changes:

- For **Bash** users:

  ```bash
  source ~/.bashrc
  ```

- For **Zsh** users:

  ```bash
  source ~/.zshrc
  ```

Now, try running `nvcc -V` again.

---

### **6. Alternative Method: Run Shell Script**

If you prefer an automated method, you can run the shell script that handles the installation. Simply execute:

```bash
./cuda-installer.sh
```

This will automatically handle the installation steps for you.

---

## **Troubleshooting**

- **Issue**: `nvcc --version` command not found.

  **Solution**: Ensure the environment variables are correctly set in your `~/.bashrc` or `~/.zshrc` and that you’ve sourced the file after modifications.

- **Issue**: CUDA Toolkit installation fails during the process.

  **Solution**: Ensure that you are not conflicting with an already installed version of CUDA. It might be necessary to uninstall previous CUDA versions using `sudo apt-get remove cuda` before reinstalling.

---

## **Graphical User Interface (GUI) Installers**

If you prefer to use a graphical interface for installation, you can download the **CUDA installer** with a GUI from the [CUDA downloads page](https://developer.nvidia.com/cuda-downloads).

---

## **Conclusion**

You have successfully installed **CUDA** on your system, enabling you to harness the power of GPU acceleration for deep learning tasks. You can now use this setup to run neural network models, optimize performance, and speed up computations.

For further instructions and resources, refer to the official [NVIDIA CUDA documentation](https://docs.nvidia.com/cuda/).

---

## **Visual Enhancements (optional)**

For a more graphical approach, include **screenshots** or **diagrams** of the steps, especially for the following sections:

- **Environment setup** (e.g., where to modify `~/.bashrc` or `~/.zshrc`).
- **Running `nvcc --version`** and **`nvidia-smi`** commands with expected outputs.
- **CUDA installation options** screen, showing the prompts during installation.

---

Happy coding! 🎉
