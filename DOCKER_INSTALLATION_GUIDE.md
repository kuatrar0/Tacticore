# 🐳 Docker Installation Guide for Windows

This guide will help you install Docker Desktop on Windows to run Tacticore in a containerized environment.

## 📋 Prerequisites

- Windows 10 64-bit: Pro, Enterprise, or Education (Build 15063 or later)
- Windows 11 64-bit: Home or Pro
- BIOS-level hardware virtualization support must be enabled in the BIOS settings
- At least 4GB RAM (8GB+ recommended)

## 🚀 Installation Steps

### Step 1: Enable WSL 2 (Recommended)

1. **Open PowerShell as Administrator**
   ```powershell
   # Right-click Start button → Windows PowerShell (Admin)
   ```

2. **Enable WSL 2**
   ```powershell
   wsl --install
   ```
   
   This will:
   - Enable the required optional components
   - Download and install the latest Linux kernel
   - Set WSL 2 as the default
   - Download and install a Linux distribution (Ubuntu by default)

3. **Restart your computer** when prompted

### Step 2: Download Docker Desktop

1. **Go to Docker Desktop download page**
   - Visit: https://www.docker.com/products/docker-desktop/

2. **Download Docker Desktop for Windows**
   - Click "Download for Windows"
   - The file will be named `Docker Desktop Installer.exe`

### Step 3: Install Docker Desktop

1. **Run the installer**
   - Double-click `Docker Desktop Installer.exe`
   - Follow the installation wizard

2. **Configuration options**
   - ✅ Use WSL 2 instead of Hyper-V (recommended)
   - ✅ Add shortcut to desktop
   - ✅ Use Windows containers (optional)

3. **Complete installation**
   - Click "Finish" when installation completes
   - Docker Desktop will start automatically

### Step 4: Verify Installation

1. **Open Command Prompt or PowerShell**
   ```cmd
   docker --version
   docker-compose --version
   ```

2. **Expected output**
   ```
   Docker version 24.0.7, build afdd53b
   Docker Compose version v2.21.0
   ```

## 🔧 Configuration

### Docker Desktop Settings

1. **Open Docker Desktop**
2. **Go to Settings (gear icon)**
3. **Recommended settings:**
   - **General**: Enable "Use WSL 2 based engine"
   - **Resources**: 
     - Memory: 4GB+ (adjust based on your system)
     - CPUs: 2+ cores
   - **WSL Integration**: Enable integration with your default WSL distro

### WSL 2 Integration

1. **In Docker Desktop Settings → Resources → WSL Integration**
2. **Enable integration with:**
   - Ubuntu (or your default WSL distro)
3. **Click "Apply & Restart"**

## 🧪 Test Docker Installation

### Quick Test

```bash
# Test Docker
docker run hello-world

# Test Docker Compose
docker-compose --version
```

### Test with Tacticore

```bash
# Navigate to your Tacticore directory
cd C:\Users\nicol\Documents\Tacticore

# Build the image
docker build -t tacticore .

# Run a quick test
docker run --rm tacticore python --version
```

## 🚨 Troubleshooting

### Common Issues

1. **"Docker is not recognized"**
   - Restart your computer after installation
   - Check if Docker Desktop is running
   - Verify PATH environment variable

2. **"WSL 2 installation is incomplete"**
   ```powershell
   # Run as Administrator
   wsl --update
   wsl --set-default-version 2
   ```

3. **"Hardware assisted virtualization and data execution protection must be enabled"**
   - Enable virtualization in BIOS
   - Enable Hyper-V in Windows Features
   - Disable Hyper-V if using WSL 2

4. **"Docker Desktop won't start"**
   - Check Windows version compatibility
   - Ensure WSL 2 is properly installed
   - Try running as Administrator

5. **Performance issues**
   - Increase Docker memory allocation
   - Use WSL 2 backend instead of Hyper-V
   - Close unnecessary applications

### BIOS Settings

If you encounter virtualization errors:

1. **Restart and enter BIOS/UEFI**
2. **Look for these settings:**
   - Intel: "Intel Virtualization Technology" → Enable
   - AMD: "AMD-V" → Enable
   - "Intel VT-d" or "AMD IOMMU" → Enable (if available)

### Windows Features

```powershell
# Run as Administrator
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
```

## 🔄 Alternative: Manual Installation

If Docker Desktop doesn't work, you can try:

### Option 1: Docker Toolbox (Legacy)
- For older Windows versions
- Uses VirtualBox instead of Hyper-V
- Download from: https://github.com/docker/toolbox/releases

### Option 2: WSL 2 + Docker Engine
```bash
# In WSL 2 Ubuntu
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

## 📞 Support

If you continue having issues:

1. **Check Docker Desktop logs**
   - Docker Desktop → Troubleshoot → Collect Logs

2. **Community support**
   - Docker Community Forums
   - Stack Overflow (tag: docker, windows)

3. **System requirements**
   - Ensure your Windows version is supported
   - Check if your hardware supports virtualization

## ✅ Next Steps

Once Docker is installed and working:

1. **Test with Tacticore**
   ```bash
   cd C:\Users\nicol\Documents\Tacticore
   docker-compose up --build
   ```

2. **Access the application**
   - Frontend: http://localhost:8501
   - Backend: http://localhost:8000

3. **Share with your project partner**
   - They can now run the same Docker setup
   - No need to worry about Python versions or dependencies
