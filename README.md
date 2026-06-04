<div align="center">

# Formula Manipal Driverless — EC2 Workstation Documentation

**AWS GPU Workstation · CARLA 0.9.16 · EUFS Sim 2 · ROS 2 Humble · Foxglove**

</div>

---

## Table of Contents

- [A. Starting a DCV Session](#a-starting-a-dcv-session)
- [B. Our Workspace](#b-our-workspace)
  - [Desktop Overview](#desktop-overview)
  - [Workspace Folder Structure](#workspace-folder-structure)
  - [FM DV Race Control](#fm-dv-race-control-carla-interface)
  - [EUFS2 Sim Launcher](#eufs2-sim-launcher)
- [C. Available Applications](#c-available-applications)
- [D. Initial System Setup](#d-initial-system-setup)
- [E. EUFS Sim 2 + Foxglove Installation](#e-eufs-sim-2--foxglove-installation)
- [F. CARLA Simulator Installation](#f-carla-simulator-installation)

---

## Instance Quick Reference

| Field | Value |
|---|---|
| **EC2 Public IP** | `13.204.88.75` |
| **DCV Web UI** | `https://13.204.88.75:8443` |
| **SSH User** | `ubuntu` |
| **OS** | Ubuntu 22.04 LTS |
| **GPU** | NVIDIA (driver 550) |
| **CARLA Version** | 0.9.16 |
| **ROS 2 Version** | Humble |

---

## A. Starting a DCV Session

NICE DCV provides a GPU-accelerated remote desktop to the EC2 instance. Follow these steps after every reboot or fresh login.

### Step 1 — Fix PEM File Permissions (Windows Only)

Run the following in an **Administrator Command Prompt**:

```cmd
icacls C:\path\to\workstation.pem /inheritance:r
icacls C:\path\to\workstation.pem /grant:r %USERNAME%:R

:: (Optional) Verify
icacls C:\path\to\workstation.pem
```

> **Linux/macOS:** `chmod 400 workstation.pem`

### Step 2 — SSH Into the Instance

```bash
ssh -i C:\path\to\workstation.pem ubuntu@<EC2_PUBLIC_IP>

# Current instance:
ssh -i C:\Keys\workstation.pem ubuntu@13.204.88.75
```

### Step 3 — Start / Verify DCV Server

```bash
sudo systemctl start dcvserver
sudo systemctl enable dcvserver
sudo systemctl status dcvserver

# List sessions
dcv list-sessions

# Create a session if none exists
sudo dcv create-session --owner=ubuntu --type=virtual my-session
```

| Credential | Value |
|---|---|
| **DCV Username** | `ubuntu` |
| **DCV Password** | `fmdv2026` |

### Step 4 — Connect to the Remote Desktop

Open a browser or the NICE DCV Client and navigate to:

```
https://13.204.88.75:8443/#my-session
```

> Accept the self-signed TLS certificate warning — this is expected for an internal workstation.

---

## B. Our Workspace

The remote desktop is an Xfce environment running on Ubuntu.

### Desktop Overview

![Desktop](images/desktop.png)
*The remote desktop. FM DV Race Control (top-left) and EUFS2 Sim Launcher shortcuts are visible. The taskbar at the bottom contains quick-launch icons.*

#### Desktop Taskbar — Quick Access Items

| # | Item | Description |
|---|---|---|
| 1 | Show Desktop | Minimise all windows |
| 2 | Search / Find Menu | Launch any app by name |
| 3 | Ubuntu Terminal | Bash terminal in `~/` |
| 4 | Folders | File manager |
| 5 | FM DV Race Control | Launches CARLA + Race Control panel |
| 6 | EUFS2 Sim Launcher | Launches EUFS2 + Foxglove |
| 7 | Google Chrome | Pre-logged-in browser |
| 8 | VS Code | Code editor |
| 9 | Foxglove Studio | ROS 2 visualisation |
| 10 | Blender | 3D modelling for car assets |
| 11 | Files | File manager shortcut |

### Workspace Folder Structure

![Workspace](images/workspace.png)
*`/home/ubuntu/Workspace/` folder layout*

| Folder | Contents / Purpose |
|---|---|
| `Car Model` | FBX and STL files for the EUFS car and Formula Manipal car |
| `Carla Scripts` | Python scripts for CARLA. Includes a sample compliant controller |
| `EUFS2 Scripts` | Python scripts for EUFS Sim 2. Includes a sample MPC script with all ROS topics |
| `EUFS Scripts` | Python scripts for the original EUFS simulator |
| `Tracks` | `csv_to_waypoints.py` · `plot_waypoints.py` · `transform_cones.py` (WIP) |
| `EUFS Tracks` | All known track cone coordinates and corresponding waypoints |
| `Work` | Sub-folders: **Perception**, **Path Planning**, **SLAM**, **Simulation** |

> ⚠️ The `~/Launchers/` folder must **not** be modified — it contains the one-click launcher scripts for CARLA and EUFS2.

#### Tracks Folder Scripts

- **`csv_to_waypoints.py`** — Converts any cone-coordinates CSV to a waypoints CSV for the car to follow.
- **`plot_waypoints.py`** — Visualises waypoints and track layout.
- **`transform_cones.py`** — Converts cone-coordinates CSV into a format EUFS2 accepts *(still a WIP)*.

### FM DV Race Control (CARLA Interface)

![FM DV Race Control](images/carla.png)
*FM DV Race Control panel alongside CARLA running the Town10HD_Opt map*

The **FM DV Race Control** shortcut launches CARLA together with a custom control panel:

- A directory browser to select scripts, waypoints, and cone-coordinate CSVs.
- Buttons to **Spawn Track**, **Run**, **Reset**, and **Stop** the simulation.
- An **Edit Values** panel to tune algorithm parameters at runtime without editing code.
- A live console showing timestamped output from CARLA and the controller.

> Controller scripts must follow the specific format expected by Race Control. A compliant sample script is provided in `Carla Scripts/`.

### EUFS2 Sim Launcher

![Foxglove Studio](images/foxglove.png)
*Foxglove Studio connected to EUFS2 via WebSocket (`ws://localhost:8765`). Left panel shows ROS topics; right panel shows the 3D cone/track visualisation.*

The **EUFS2 Sim Launcher** opens three terminals simultaneously:

| Terminal | Process | Notes |
|---|---|---|
| 1 | EUFS Sim 2 | Full ROS 2 simulation environment |
| 2 | Foxglove Bridge | Exposes ROS topics over WebSocket on port 8765 |
| 3 | Foxglove Studio | GPU-accelerated visualisation UI |

Once Foxglove opens, click **Open Connection → Foxglove WebSocket → Open** and confirm the URL is `ws://localhost:8765`. Then run your car controller script in a separate terminal.

---

## C. Available Applications

| Application | Description |
|---|---|
| **Google Chrome** | Logged in as `fmdriverless26@gmail.com`. Used for GitHub and Google Drive. |
| **VS Code** | Launch with `code .` from any terminal to open in that directory. |
| **Foxglove Studio** | ROS 2 data visualisation. Connect to `ws://localhost:8765`. |
| **Gazebo** | Alternative robotics simulation and visualisation environment. |
| **Blender** | Create/edit skeletal rigs and FBX car models for Unreal Engine 4 (CARLA). |

### Credentials

| | Value |
|---|---|
| **GitHub Username** | `fmdriverless26` |
| **GitHub Token (PAT)** | `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` |
| **Gmail Account** | `fmdriverless26@gmail.com` |

> Use the PAT as your password when Git prompts for credentials over HTTPS.

---

## D. Initial System Setup

> These steps are only needed when provisioning a **fresh Ubuntu EC2 instance**.

### Step 1 — Install Essential Packages

```bash
sudo apt install -y \
    git curl wget build-essential cmake \
    python3-pip python3-venv software-properties-common \
    htop unzip net-tools
```

### Step 2 — Install NVIDIA GPU Driver

```bash
sudo apt install -y nvidia-driver-550
sudo reboot

# After reboot — verify
nvidia-smi

# Set NVIDIA as default OpenGL vendor
echo 'export __GLX_VENDOR_LIBRARY_NAME=nvidia' >> ~/.bashrc
source ~/.bashrc
```

### Step 3 — One-Shot Application Install Script

Save as `install_apps.sh` and run with `bash install_apps.sh`.

```bash
#!/bin/bash

# Google Chrome
wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install -y /tmp/chrome.deb && rm /tmp/chrome.deb

# VS Code
sudo snap install code --classic

# Blender
sudo snap install blender --classic

# LibreOffice
sudo apt install -y libreoffice

# Gazebo 11
sudo apt install -y gazebo11 libgazebo11-dev

# Build tools & utilities
sudo apt install -y htop git curl wget build-essential cmake python3-pip

# Extra tools
sudo apt install -y \
    ncdu filelight bleachbit rclone p7zip-full \
    gimp flameshot simplescreenrecorder

# NVIDIA driver
sudo apt install -y nvidia-driver-550
echo 'export __GLX_VENDOR_LIBRARY_NAME=nvidia' >> ~/.bashrc
source ~/.bashrc

echo "All applications installed."
```

### Step 4 — Verify Installations

```bash
google-chrome --version
code --version
blender --version
libreoffice --version
gazebo --version
nvidia-smi
```

---

## E. EUFS Sim 2 + Foxglove Installation

### 1. Install ROS 2 Humble

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository universe -y
sudo apt update

# Add ROS 2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS 2 apt repository
echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions python3-rosdep
```

### 2. Set Up the EUFS Sim 2 Workspace

```bash
mkdir -p ~/modified_eufs_sim_v2 && cd ~/modified_eufs_sim_v2
git clone https://github.com/formula-manipal-driverless/modified_eufs_sim_v2.git
cd modified_eufs_sim_v2
```

### 3. Install Dependencies with rosdep

```bash
cd ~/modified_eufs_sim_v2
sudo rosdep init    # ignore 'already exists' warning
rosdep update
rosdep install --from-paths . --ignore-src -r -y
```

### 4. Build the Workspace

```bash
cd ~/modified_eufs_sim_v2/eufs-master
colcon build --symlink-install
```

### 5. Source the Workspace

```bash
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
echo 'source ~/modified_eufs_sim_v2/eufs-master/install/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

### 6. Install Foxglove Bridge

```bash
sudo apt update
sudo apt install -y ros-humble-foxglove-bridge
```

### 7. Install Foxglove Studio

```bash
sudo snap install foxglove-studio
```

### 8. (Optional) Install EUFS CLI

```bash
cd ~/modified_eufs_sim_v2/eufs_sim/eufs_cli
pip install --user -e ./eufscli
pip install --user -e ./eufsinstall
pip install --user -e ./eufssim

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

eufs update
```

### 9. Launch EUFS Sim 2 — Terminal 1

```bash
source ~/modified_eufs_sim_v2/eufs-master/install/setup.bash
ros2 launch eufs_sim2 eufs_sim2.launch.py

# — or with CLI —
eufs sim run
```

### 10. Launch Foxglove Bridge — Terminal 2

```bash
source ~/modified_eufs_sim_v2/eufs-master/install/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml use_foxglove_whitelist:=False
```

### 11. Launch Foxglove Studio — Terminal 3

```bash
__GLX_VENDOR_LIBRARY_NAME=nvidia foxglove-studio \
    --enable-gpu-rasterization --ignore-gpu-blocklist --use-angle=gl
```

Connect to `ws://localhost:8765` inside Foxglove Studio.

### 12. (Optional) Set NVIDIA as Default OpenGL Vendor

```bash
echo 'export __GLX_VENDOR_LIBRARY_NAME=nvidia' >> ~/.bashrc
source ~/.bashrc
```

### One-Click Launcher Script

Save as `~/Launchers/EUFS2/run_eufs2.py` and make it executable with `chmod +x ~/Launchers/EUFS2/run_eufs2.py`.

```python
#!/usr/bin/env python3
import subprocess, time, os, signal, sys

workspace_path = "/home/ubuntu/modified_eufs_sim_v2/eufs_sim/eufs-master"
cli_home       = "/home/ubuntu/modified_eufs_sim_v2/eufs_sim/eufs_cli"
processes = []

def signal_handler(sig, frame):
    print("\nShutting down all terminals...")
    for p in processes:
        try: p.terminate()
        except: pass
    sys.exit(0)

def launch_terminal(cmd):
    full_cmd = (
        f"export EUFS_MASTER={workspace_path} && "
        f"export EUFS_CLI_HOME={cli_home} && "
        "export PATH=\"$HOME/.local/bin:$PATH\" && "
        "source /opt/ros/humble/setup.bash && "
        f"source {workspace_path}/install/setup.bash && "
        f"{cmd}; exec bash"
    )
    proc = subprocess.Popen(
        ['gnome-terminal', '--', 'bash', '-c', full_cmd],
        preexec_fn=os.setsid
    )
    processes.append(proc)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    launch_terminal("eufs sim run")
    time.sleep(5)
    launch_terminal("ros2 launch foxglove_bridge foxglove_bridge_launch.xml use_foxglove_whitelist:=False")
    time.sleep(3)
    launch_terminal(
        "__GLX_VENDOR_LIBRARY_NAME=nvidia foxglove-studio "
        "--enable-gpu-rasterization --ignore-gpu-blocklist --use-angle=gl"
    )
    print("All launched. Press Ctrl+C to close all windows.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass
```

---

## F. CARLA Simulator Installation

### Step 1 — Download and Extract CARLA

Download `CARLA_0.9.16.tar.gz` from the [official releases page](https://github.com/carla-simulator/carla/releases).

```bash
tar -xzf CARLA_0.9.16.tar.gz -C ~/
mv ~/CARLA_0.9.16 ~/Carla
```

### Step 2 — Make the Launcher Executable

```bash
cd ~/Carla
chmod +x CarlaUE4.sh
```

### Step 3 — Run CARLA

```bash
# Default launch
./CarlaUE4.sh

# High quality, windowed 1600×900
./CarlaUE4.sh -quality-level=High -ResX=1600 -ResY=900 -windowed

# Epic quality, windowed 1920×1080
./CarlaUE4.sh -quality-level=Epic -ResX=1920 -ResY=1080 -windowed
```

> Full CARLA documentation: https://carla.readthedocs.io/en/latest/

---

<div align="center">

*Formula Manipal Driverless · EC2 Workstation Documentation · v1.0 · June 2026*

</div>
