# 🤝 Project Sharing Guide for Tacticore

This guide explains how to share your Tacticore project with your project partner, ensuring it works consistently across different environments.

## 🎯 Sharing Options

### Option 1: Docker (Recommended - Most Reliable)

**Why Docker is best for sharing:**
- ✅ Identical environment across all machines
- ✅ No Python version conflicts
- ✅ No dependency issues
- ✅ Works on Windows, Mac, and Linux
- ✅ Easy to set up and run

**What your partner needs:**
1. Docker Desktop installed
2. Your project files
3. Run `docker-compose up --build`

**Setup for your partner:**
```bash
# 1. Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop/

# 2. Clone your repository
git clone <your-repo-url>
cd Tacticore

# 3. Run the application
docker-compose up --build

# 4. Access the app
# Frontend: http://localhost:8501
# Backend: http://localhost:8000
```

### Option 2: Cross-Platform Scripts (Fallback)

If Docker doesn't work, use the cross-platform setup scripts:

**Windows:**
```cmd
scripts\setup_cross_platform.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/setup_cross_platform.sh
./scripts/setup_cross_platform.sh
```

## 📦 What to Share

### Essential Files (Always include)
```
Tacticore/
├── src/                          # Source code
├── maps/                         # Map data
├── dataset/                      # Demo files and parsed data
├── src/backend/models/           # Trained ML models
├── requirements.txt              # Python dependencies
├── requirements-docker.txt       # Docker dependencies
├── Dockerfile                    # Docker configuration
├── docker-compose.yml           # Docker services
├── docker-compose.dev.yml       # Development mode
├── .dockerignore                # Docker ignore rules
├── docker-entrypoint.sh         # Startup script
├── scripts/                     # Setup scripts
├── README.md                    # Main documentation
├── DOCKER_SETUP.md             # Docker instructions
├── DOCKER_INSTALLATION_GUIDE.md # Docker installation
└── PROJECT_SHARING_GUIDE.md    # This file
```

### Optional Files (Include if available)
```
├── .venv/                       # Virtual environment (if small)
├── results/                     # Analysis results
└── tests/                       # Test files
```

## 🚀 Quick Start for Your Partner

### Method 1: Docker (Recommended)

1. **Send them this message:**
   ```
   Hi! Here's how to run the Tacticore project:
   
   1. Install Docker Desktop: https://docker.com/products/docker-desktop/
   2. Clone the repo: git clone <repo-url>
   3. Run: docker-compose up --build
   4. Open: http://localhost:8501
   
   That's it! Everything should work identically to my setup.
   ```

2. **If they have issues:**
   - Send them `DOCKER_INSTALLATION_GUIDE.md`
   - Check `DOCKER_SETUP.md` for troubleshooting

### Method 2: Manual Setup (Fallback)

1. **Send them this message:**
   ```
   If Docker doesn't work, try the manual setup:
   
   Windows: Run scripts\setup_cross_platform.bat
   Linux/Mac: Run ./scripts/setup_cross_platform.sh
   
   Then: streamlit run src/streamlit_app/app.py
   ```

## 🔧 Pre-Sharing Checklist

Before sharing, make sure:

- [ ] **All files are committed to Git**
  ```bash
  git add .
  git commit -m "Ready for sharing"
  git push
  ```

- [ ] **Docker setup is tested**
  ```bash
  docker-compose up --build
  # Verify both frontend and backend work
  ```

- [ ] **Documentation is complete**
  - README.md has clear instructions
  - DOCKER_SETUP.md is comprehensive
  - All setup scripts are included

- [ ] **Dependencies are pinned**
  - requirements.txt has specific versions
  - requirements-docker.txt is complete

- [ ] **Data is included**
  - Sample demo files in dataset/
  - Trained models in src/backend/models/
  - Map data in maps/

## 🐛 Common Issues and Solutions

### Issue: "Docker not found"
**Solution:** Install Docker Desktop and restart computer

### Issue: "Port already in use"
**Solution:** 
```bash
# Kill processes using ports 8501/8000
netstat -tulpn | grep :8501
netstat -tulpn | grep :8000
# Or change ports in docker-compose.yml
```

### Issue: "Permission denied" (Linux/Mac)
**Solution:**
```bash
chmod +x scripts/*.sh
chmod +x docker-entrypoint.sh
```

### Issue: "Python version mismatch"
**Solution:** Use Docker instead of manual setup

### Issue: "Missing dependencies"
**Solution:** 
```bash
# Rebuild Docker image
docker-compose down
docker-compose up --build
```

## 📋 Testing Checklist

Before sharing, test on a clean system:

- [ ] **Fresh Windows machine**
  - Install Docker Desktop
  - Clone repository
  - Run `docker-compose up --build`
  - Verify frontend and backend work

- [ ] **Different Python version**
  - Test with Python 3.8, 3.9, 3.10, 3.11
  - Use cross-platform setup scripts

- [ ] **Different operating system**
  - Test on Windows, Mac, Linux
  - Verify all features work

## 🎯 Best Practices

### For You (Project Owner)
1. **Use Docker for development** - ensures consistency
2. **Pin all dependencies** - prevents version conflicts
3. **Test on clean systems** - catch issues early
4. **Document everything** - make setup foolproof
5. **Include sample data** - helps with testing

### For Your Partner
1. **Try Docker first** - most reliable option
2. **Follow setup guides** - don't skip steps
3. **Check system requirements** - ensure compatibility
4. **Report issues early** - include error messages
5. **Use provided scripts** - don't reinvent the wheel

## 📞 Support

If your partner has issues:

1. **Check the logs**
   ```bash
   docker-compose logs -f
   ```

2. **Verify system requirements**
   - Windows 10/11 with WSL 2
   - 4GB+ RAM
   - Docker Desktop installed

3. **Try alternative setup**
   - Use cross-platform scripts
   - Manual Python setup

4. **Share error messages**
   - Include full error output
   - Mention operating system
   - Include Docker version

## 🎉 Success!

Once your partner can run the project:

- ✅ Frontend accessible at http://localhost:8501
- ✅ Backend API working at http://localhost:8000
- ✅ Can upload and analyze demo files
- ✅ ML model predictions working
- ✅ All features functional

**Congratulations! Your project is now truly cross-platform and shareable!** 🚀
