#!/bin/bash

# Quick Setup Script for Gradio App Deployment
# This script prepares example images and verifies everything is ready

echo "======================================================================"
echo "CHEST X-RAY APP - DEPLOYMENT PREPARATION"
echo "======================================================================"

# Navigate to project directory
cd ~/Downloads/med-image/chest-xray-project

echo -e "\n1. Creating examples directory..."
mkdir -p examples

echo -e "\n2. Copying example X-rays from dataset..."

# Copy 2 normal examples
cp ../chest_xray_organized/test/NORMAL/IM-0001-0001.jpeg examples/normal1.jpeg 2>/dev/null || \
cp ../chest_xray_organized/test/NORMAL/IM-0003-0001.jpeg examples/normal1.jpeg 2>/dev/null || \
cp $(ls ../chest_xray_organized/test/NORMAL/*.jpeg | head -1) examples/normal1.jpeg

cp ../chest_xray_organized/test/NORMAL/IM-0005-0001.jpeg examples/normal2.jpeg 2>/dev/null || \
cp $(ls ../chest_xray_organized/test/NORMAL/*.jpeg | tail -1) examples/normal2.jpeg

# Copy 2 pneumonia examples  
cp ../chest_xray_organized/test/PNEUMONIA/person1_bacteria_1.jpeg examples/pneumonia1.jpeg 2>/dev/null || \
cp $(ls ../chest_xray_organized/test/PNEUMONIA/*.jpeg | head -1) examples/pneumonia1.jpeg

cp ../chest_xray_organized/test/PNEUMONIA/person2_bacteria_2.jpeg examples/pneumonia2.jpeg 2>/dev/null || \
cp $(ls ../chest_xray_organized/test/PNEUMONIA/*.jpeg | tail -1) examples/pneumonia2.jpeg

echo "✓ Example images copied"

echo -e "\n3. Checking files..."
echo "App files:"
ls -lh app.py requirements_deployment.txt README_SPACE.md 2>/dev/null && echo "✓ All app files present" || echo "✗ Missing app files"

echo -e "\nModel file:"
ls -lh models/best_model.pth && echo "✓ Model file present" || echo "✗ Model file missing!"

echo -e "\nExample images:"
ls -lh examples/*.jpeg && echo "✓ Example images present" || echo "✗ Example images missing"

echo -e "\n======================================================================"
echo "SETUP SUMMARY"
echo "======================================================================"
echo ""
echo "Your app is ready for:"
echo "  1. Local testing: python app.py"
echo "  2. Hugging Face deployment"
echo ""
echo "Next steps:"
echo "  - Test locally first: cd ~/Downloads/med-image/chest-xray-project && python app.py"
echo "  - Follow DEPLOYMENT_GUIDE.md for Hugging Face deployment"
echo ""
echo "======================================================================"
