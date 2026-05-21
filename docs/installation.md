# Installation

Optional detector virtual environments each ship their own PyTorch CUDA build in an isolated venv (for example **MMPose / Py-Feat / SPIGA** use **`cu118`**, **SAM 3D Body** uses **`cu121`**, **WhisperX** uses **`cu126`**). They do not share one global PyTorch install, so different CUDA lines in different venvs do not conflict. **SAM 3D Body** expects a **CUDA 12.1 toolkit** with matching **`nvcc`** when **Detectron2** is built from source.

<!-- TOC -->
- [Installation](#installation)
  - [Docker](#docker)
  - [Prerequisites](#prerequisites)
    - [Python 3.10](#python-310)
    - [Conda](#conda)
    - [Cuda 11.8](#cuda-118)
    - [SAM 3D Body and Detectron2 (CUDA toolkit 12.1)](#sam-3d-body-and-detectron2-cuda-toolkit-121)
    - [FFmpeg](#ffmpeg)
    - [Git](#git)
    - [On Windows: Microsoft Visual C++](#on-windows-microsoft-visual-c)
    - [On Windows: make](#on-windows-make)
    - [On Windows: Enable long path support](#on-windows-enable-long-path-support)
  - [Clone the repository](#clone-the-repository)
  - [Makefile installation](#makefile-installation)
  - [Additional notes](#additional-notes)
<!-- TOC -->

## Docker 

You can install NICE Toolbox using Docker. With Docker, you won't need to install dependencies manually as they are prepackaged into the Docker image.

To download and run the latest NICE Toolbox Docker image:

```shell
docker pull mpioslab/nicetoolbox
docker run --rm --gpus all -it mpioslab/nicetoolbox
```

It should start an interactive bash session inside a Docker container.  Follow [getting started](https://nicetoolbox.readthedocs.io/en/stable/getting_started.html) to enable a virtual environment and run detectors.

## Prerequisites

### Python 3.10

Please find the download links under the [official python](https://www.python.org/downloads/) pages. The latest installer of a stable release of Python 3.10.10 can be downloaded [from here](https://www.python.org/downloads/release/python-31011/).

If you are a Windows user, please add python to your `PATH` variable as explained on [educative.io](https://www.educative.io/answers/how-to-add-python-to-path-variable-in-windows).

For Windows users, we also recommend clicking **"Disable path length limit"** on the final screen of the Python installer. This removes the default 260-character path limit that can cause failures with long dataset or session names. 

![python_windows_path_limit.png](graphics/python_windows_path_limit.png)

You can always disable or enable line limit manually. See [Enable long path support](#on-windows-enable-long-path-support) for more details.

### Conda

Conda can be installed through different installers, see [conda.io](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). A popular one is the Anaconda Distribution -- it uses the anaconda channel by default which is subject to specific licensing.
An alternative option is the [Miniforge](https://github.com/conda-forge/miniforge) installer, which uses the [conda-forge](https://conda-forge.org/) channel and comes with open-source packages.
Please find instructions to install Miniforge on their [official website](https://github.com/conda-forge/miniforge).

If you installed Conda through Anaconda, you can switch to the free conda-forge channel following these steps:

```bash
# check what is currently set
conda config --show channels

# remove all channels other than conda-forge
conda config --remove channels defaults

# add conda-forge if not already present
conda config --add channels conda-forge
```

```{important}
During the installation of Conda, it is **crucial not to select** the option to register Conda's Python as the default Python interpreter.
This is because the Nice Toolbox requires **Python version 3.10** to be set as the default.

![miniforge_default_python.png](graphics/miniforge_default_python.png)

Selecting this option during installation may result in errors or conflicts, as Conda's Python version may differ from the required version for NiceToolbox. To ensure proper functionality, make sure Python 3.10 remains your default version.
```

```{important}
On Windows, after installing Conda, ensure that the Conda paths are added to the SYSTEM environment variables. For details see: https://saturncloud.io/blog/solving-the-conda-command-not-recognized-issue-on-windows-10/#step-2-add-conda-to-the-path
```

### Cuda 11.8

Several Makefile targets install PyTorch from `https://download.pytorch.org/whl/cu118` (for example Multiview ETH-XGaze, Py-Feat, SPIGA, and the OpenMMLab script for MMPose). Those wheels bundle a CUDA 11.8 **runtime** for inference; you normally do **not** need the full CUDA toolkit unless you compile CUDA extensions yourself.

Please find installation instructions on the official websites: for [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [Linux Ubuntu](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

### SAM 3D Body and Detectron2 (CUDA toolkit 12.1)

Upstream **SAM 3D Body** code lives in the git submodule **`submodules/sam-3d-body`** ([`OSLabTools/sam-3d-body`](https://github.com/OSLabTools/sam-3d-body)). After cloning, run **`git submodule update --init submodules/sam-3d-body`** (or clone with **`--recurse-submodules`**) so that directory contains the **`sam_3d_body/`** package.

The SAM 3D Body virtual environment installs **`torch==2.4.1+cu121`** and **`torchvision==0.19.1+cu121`** (see `nicetoolbox/detectors/method_detectors/sam_3d_body/sam_3d_body_pip_requirements.txt`). **Detectron2** is required and is installed by **`make install_sam3d_body`**, which tries **building from the GitHub repository** first. That compile step uses **`nvcc`** from the **CUDA toolkit** on your system; it must match PyTorch’s CUDA (**12.1**), or the build fails with a CUDA version mismatch.

**GPU driver:** `nvidia-smi` must report a driver whose maximum CUDA version is **at least 12.1**. PyTorch **`cu121`** wheels bundle a CUDA 12.1 **runtime**; the driver only needs to be new enough to run that runtime. Building Detectron2 from source additionally needs a **CUDA 12.1 toolkit** so **`nvcc`** matches.

**Checklist for a successful Linux install**

1. **Driver:** `nvidia-smi` should show a maximum CUDA version **≥ 12.1** (the line labeled “CUDA Version” is the driver capability, not your installed toolkit).

2. **CUDA toolkit 12.1 for `nvcc`:** Install the [CUDA 12.1 toolkit](https://developer.nvidia.com/cuda-12-1-0-download-archive) (or your administrator’s equivalent). Set **`CUDA_HOME`** to that install and put **`${CUDA_HOME}/bin`** early on **`PATH`** so **`nvcc --version`** reports **release 12.1**.

   **Shell / `PATH`:** Editing **`~/.bashrc`** does not affect shells that are already open—**`source ~/.bashrc`** or open a new terminal. **Conda** **`activate.d`** scripts and **environment modules** can point **`nvcc`** at a different toolkit (for example **12.6** under **`/is/software/nvidia/cuda-12.6`** when that path is missing on your machine). Check with **`which nvcc`** and **`nvcc -V`** in the same shell you use for **`make`**.

3. **Host C++ compiler:** Detectron2’s CUDA extensions are compiled with **`nvcc`**, which only supports certain **GCC** versions per toolkit. For CUDA **12.1**, **GCC must not be newer than 12** (GCC **13+** typically triggers `unsupported GNU version! gcc versions later than 12 are not supported` in `host_config.h`). Install an older GCC (e.g. on Ubuntu **24.04**: `gcc-12` / `g++-12`) and **only for the build session** point the build at it, for example:

   ```bash
   export CC=/usr/bin/gcc-12
   export CXX=/usr/bin/g++-12
   ```

4. **Run the Makefile** from the repo root:

   ```bash
   make install_sam3d_body
   ```

5. **Verify** PyTorch in that venv:

   ```bash
   ./envs/sam_3d_body/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```

**If the git build fails:** the Makefile installs from Meta’s wheel index next (`cu121` / `torch2.4`); if **`pip`** reports **no matching distribution**, the index may be empty or blocked on your network—in that case fix **GCC/toolkit** and retry the source build, or install Detectron2 manually per **`sam_3d_body_pip_requirements.txt`** comments.

**Windows:** Install **CUDA Toolkit 12.1** and a **Visual Studio** toolchain supported by that CUDA version; set **`CUDA_PATH`** per NVIDIA’s [Windows guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html), then run **`make install_sam3d_body`**.

**Alternative:** If you cannot satisfy toolkit **12.1** + compiler constraints, you can repin SAM to **`cu118`** and the matching Detectron2 wheel URL (see comments in `sam_3d_body_pip_requirements.txt` and the Makefile)—that aligns SAM with the same PyTorch CUDA line as several other detectors, at the cost of editing pins and retesting.

**WhisperX (`cu126`):** **`make install_whisperx`** installs **PyTorch `cu126`** into **`./envs/whisperx`**, a separate venv. That does not affect **`./envs/sam_3d_body`**. You only need a driver new enough for the **highest** CUDA line you actually run (check **`nvidia-smi`** if you use both detectors on one GPU).

### FFmpeg

On Linux Ubuntu, please find detailed instructions [here](https://phoenixnap.com/kb/install-ffmpeg-ubuntu).

On Windows, you can follow [phoenixnap.com](https://phoenixnap.com/kb/ffmpeg-windows):

1. Visit the official [FFmpeg website](https://ffmpeg.org/download.html) to get the latest version
of the FFmpeg package and binary files.
2. Hover over the Windows icon with your mouse and click on 'Windows builds from gyan.dev'
3. This redirects you to a page having FFmpeg binaries. Install the latest git master branch build,
e.g., ffmpeg-git-essentials.7z.
4. Extract the downloaded files and rename the extracted folder as ffmpeg.
5. Move the folder to the root of the C drive or the folder of your choice.
6. Add FFmpeg to `PATH` in Windows SYSTEM environment variables.

### Git 

Ensure that Git is installed on your system. You can find installation instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

### On Windows: Microsoft Visual C++

Microsoft Visual C++ 14.0 or greater is required for compiling some of the dependencies. Get it with [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

### On Windows: Make

Nice Toolbox uses Makefiles for simple installation process. Follow these steps to install `make` on Windows for use with **Git Bash**:

**Step 1:** Download `make` for Windows
- Go to the official **ezwinports** SourceForge page:  
   🔗 [https://sourceforge.net/projects/ezwinports/files/](https://sourceforge.net/projects/ezwinports/files/)
- Download the latest version of **make**:  
   - Look for a file named:  `make-<latest_version>-without-guile-w32-bin.zip`

**Step 2:** Extract the ZIP File
- Unzip the downloaded `make-<latest_version>-without-guile-w32-bin.zip` file.

**Step 3:** Copy the Files to Git Bash’s MinGW64 Folder
- Navigate to: `C:\Program Files\Git\mingw64`
- Copy the contents of the extracted folder (copy all folders) into `C:\Program Files\Git\mingw64`. 
- **IMPORTANT:** Do NOT overwrite or replace any existing files.

**Note:**  
After copying the files, you must **restart Git Bash** for the changes to take effect.


### On Windows: Enable long path support

We always recommend to enable long path support. If you did not click **"Disable path length limit"** during Python installation, you can enable long path support manually via **PowerShell** command line:

1. Search for **PowerShell** in the Start menu, right-click it and select **Run as administrator**.
2. Run the following command:

```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
```

See [Microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation) for details.

## Clone the repository

Clone the NICE Toolbox repository and navigate to its directory:

```bash
git clone --recurse-submodules git@github.com:OSLabTools/nicetoolbox.git
cd /path/to/nicetoolbox
```

The `--recurse-submodules` flag ensures that all submodules are are automatically initialized and updated (including **MMPose**, **SPIGA**, **SAM 3D Body** at `submodules/sam-3d-body`, etc.). 
Alternatively, you can run the following commands after having cloned the repository without this flag:
```bash
git submodule init           # to initialize your local configuration file
git submodule update         # to fetch all the files from the submodules and check out the appropriate commit
git submodule update --init  # to combine the git submodule init and git submodule update steps
```

## Makefile installation

The NICE Toolbox includes a Makefile that handles the installation of all required libraries and dependencies. It also downloads assets and an example dataset, and generates configuration files. Available commands include:

- `make` or `make all`  - Run all the commands below.
- `make create_machine_specifics` - Generate the machine-specific configuration file.
- `make create_project` - Generate the project configuration file.
- `make install` - Install all dependencies. On Linux, if **`./envs/sam_3d_body`** is missing, the Makefile also runs **`install_sam3d_body`** (PyTorch **`cu121`** + **Detectron2**); that step needs a matching **CUDA 12.1 toolkit / `nvcc`**—see [SAM 3D Body and Detectron2](#sam-3d-body-and-detectron2-cuda-toolkit-121).
- `make download_assets` - Check and download assets.
- `make download_dataset` - Check and download the example dataset.

```{note}
Conda is required for installing the OpenMMLab environment (human pose estimation framework).
If you need to use different versions of Python or CUDA, you can adjust the relevant lines in the `Makefile` accordingly.
The order of specific make commands listed above is essential due to iterated dependencies.
```

In case of errors during installation, you can run `make clean_all` to remove all virtual environments. After that, you can restart the installation.

### On Linux

Open a **terminal** (on Linux) or **Git Bash** (on Windows) and navigate to the directory of the repository, then run the command `make`:

```bash
cd /path/to/nicetoolbox/
make        
```

## Additional notes

Please check [rerun privacy policies](https://www.rerun.io/privacy).
Although rerun.io is used in local mode, the application will be collecting user information. To disable these analytics, activate the code environment in `env/` and then run:

```bash
rerun analytics config   ##to see current configuration
rerun analytics disable
rerun analytics config   ## to check if the change is applied
```
