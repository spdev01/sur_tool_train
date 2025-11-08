#!/bin/bash
# filepath: setup_runpod.sh

set -e  # Exit on error

echo "=========================================="
echo "RunPod YOLO Training Environment Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# 1. Update system packages
echo ""
echo "Step 1: Updating system packages..."
apt-get update -qq
apt-get install -y zip unzip git wget curl screen tmux htop -qq
print_status "System packages updated"

# 2. Check Python and CUDA
echo ""
echo "Step 2: Checking Python and CUDA..."
python_version=$(python --version 2>&1)
print_status "Python: $python_version"

if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//' 2>/dev/null || echo "N/A")
    print_status "GPU: $gpu_info"
    print_status "CUDA: $cuda_version"
else
    print_error "NVIDIA GPU not detected!"
    exit 1
fi

# 3. Install Python dependencies
echo ""
echo "Step 3: Installing Python packages..."
pip install -q --upgrade pip
pip install -q ultralytics
pip install -q onnx onnxruntime onnxruntime-gpu
pip install -q opencv-python
pip install -q PyYAML
print_status "Python packages installed"

# 4. Verify PyTorch and CUDA
echo ""
echo "Step 4: Verifying PyTorch installation..."
python << 'PYEOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
PYEOF
print_status "PyTorch verified"

# 5. Create project directory structure
echo ""
echo "Step 5: Setting up project directory..."
PROJECT_DIR="/workspace/sur_tool_train"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# # 6. Clone repository
# echo ""
# echo "Step 6: Cloning repository..."
# read -p "Enter your GitHub repository URL (or press Enter to skip): " REPO_URL

# if [ ! -z "$REPO_URL" ]; then
#     # Extract repo name from URL
#     REPO_NAME=$(basename "$REPO_URL" .git)
    
#     if [ -d "$REPO_NAME" ]; then
#         print_warning "Directory $REPO_NAME already exists. Skipping clone."
#     else
#         git clone "$REPO_URL"
#         print_status "Repository cloned successfully"
        
#         # Check if YOLO_v12_train directory exists in the repo
#         if [ -d "$REPO_NAME/YOLO_v12_train" ]; then
#             print_status "Found YOLO_v12_train directory"
#             cd "$REPO_NAME/YOLO_v12_train"
#         fi
#     fi
# else
#     print_warning "Skipping git clone. Creating empty directory structure..."
#     mkdir -p base_models
#     mkdir -p dataset/images/train
#     mkdir -p dataset/images/val
#     mkdir -p dataset/images/test
#     mkdir -p dataset/labels/train
#     mkdir -p dataset/labels/val
#     mkdir -p dataset/labels/test
#     mkdir -p runs/detect
#     print_status "Project directory created at $PROJECT_DIR"
# fi

# 7. Download YOLO model weights
echo ""
echo "Step 7: Downloading YOLO model weights..."
read -p "Download YOLO model weights? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Ensure we're in the right directory
    if [ ! -z "$REPO_URL" ] && [ -d "$PROJECT_DIR/$REPO_NAME/YOLO_v12_train" ]; then
        cd "$PROJECT_DIR/$REPO_NAME/YOLO_v12_train"
    fi
    
    mkdir -p base_models
    cd base_models
    
    # Array of model URLs
    declare -a models=(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt"
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt"
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt"
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt"
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt"
    )
    
    echo ""
    echo "Downloading models to: $(pwd)"
    echo ""
    
    for model_url in "${models[@]}"; do
        model_name=$(basename "$model_url")
        
        if [ -f "$model_name" ]; then
            print_warning "$model_name already exists. Skipping."
        else
            echo "Downloading $model_name..."
            wget -q --show-progress "$model_url"
            print_status "$model_name downloaded"
        fi
    done
    
    echo ""
    print_status "All YOLO models downloaded successfully!"
    echo ""
    echo "Available models:"
    ls -lh *.pt
    
    cd ..
else
    print_warning "Skipping model download. You'll need to upload models manually."
fi

# 8. Display next steps
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""

if [ ! -z "$REPO_URL" ]; then
    WORK_DIR="$PROJECT_DIR/$REPO_NAME/YOLO_v12_train"
else
    WORK_DIR="$PROJECT_DIR"
fi

print_status "Working directory: $WORK_DIR"
echo ""

if [ -z "$REPO_URL" ]; then
    echo "Next steps (Manual Upload):"
    echo ""
    echo "Upload your dataset using one of these methods:"
    echo ""
    echo "Method A - SCP from local machine:"
    echo "  scp -P <port> -r /local/dataset/* root@<pod-id>.runpod.io:$WORK_DIR/dataset/"
    echo ""
    echo "Method B - Jupyter Lab file browser:"
    echo "  Navigate to $WORK_DIR and drag/drop your dataset folder"
    echo ""
    echo "After uploading, verify your files:"
    echo "  ls -la $WORK_DIR/base_models/"
    echo "  ls -la $WORK_DIR/dataset/data.yaml"
else
    echo "Next steps:"
    echo ""
    echo "1. Navigate to your project:"
    echo "   cd $WORK_DIR"
    echo ""
    echo "2. Upload your dataset (if not in repo):"
    echo "   scp -P <port> -r /local/dataset/* root@<pod-id>.runpod.io:$WORK_DIR/dataset/"
    echo ""
    echo "3. Verify your files are present:"
    echo "   ls -la base_models/"
    echo "   ls -la dataset/data.yaml"
fi

echo ""
echo "Then start training:"
echo "  cd $WORK_DIR"
echo "  screen -S training"
echo "  python train.py"
echo ""
echo "  Detach from screen: Ctrl+A, then D"
echo "  Reattach to screen: screen -r training"
echo ""
echo "=========================================="