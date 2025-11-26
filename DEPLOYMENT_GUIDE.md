# 🚀 Deployment Guide for Hugging Face Spaces

This guide will walk you through deploying your Chest X-Ray Pneumonia Detection app to Hugging Face Spaces for FREE!

## 📋 Prerequisites

1. **Hugging Face Account** (Free)
   - Sign up at: https://huggingface.co/join
   - Verify your email

2. **Git Installed**
   - Check: `git --version`
   - Install from: https://git-scm.com/downloads

3. **Git LFS Installed** (for large model files)
   - Check: `git lfs version`
   - Install from: https://git-lfs.github.com/

4. **Your Trained Model**
   - `models/best_model.pth` (from training)

---

## 🎯 Method 1: Web Interface Upload (Easiest)

### Step 1: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in details:
   - **Space name:** `chest-xray-pneumonia-detector` (or your choice)
   - **License:** MIT
   - **Select SDK:** Gradio
   - **SDK version:** 4.8.0
   - **Space hardware:** CPU (Free)
   - **Visibility:** Public

4. Click **"Create Space"**

### Step 2: Upload Files

In your new Space, click **"Files"** tab, then **"Add file"** → **"Upload files"**

Upload these files:
```
✓ app.py
✓ requirements_deployment.txt (rename to requirements.txt)
✓ README_SPACE.md (rename to README.md)
✓ models/best_model.pth
```

**For example images folder:**
- Create folder: Click "Add file" → "Create a new file"
- Name it: `examples/.gitkeep`
- Upload your example X-rays to the `examples/` folder

### Step 3: Wait for Build

- Hugging Face will automatically build your app
- Watch the logs in the **"App"** tab
- Build takes ~2-3 minutes
- Your app will auto-launch when ready! 🎉

---

## 🎯 Method 2: Git & Command Line (Recommended for Larger Projects)

### Step 1: Install Git LFS

```bash
# Install Git LFS
git lfs install

# Verify installation
git lfs version
```

### Step 2: Create Space on Hugging Face

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Configure as shown in Method 1
4. Click "Create Space"

### Step 3: Clone Your Space Locally

```bash
# Clone the empty space
git clone https://huggingface.co/spaces/YOUR_USERNAME/chest-xray-pneumonia-detector

cd chest-xray-pneumonia-detector
```

### Step 4: Set Up Git LFS for Large Files

```bash
# Track large model files with Git LFS
git lfs track "*.pth"
git lfs track "*.ckpt"
git lfs track "*.bin"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"
```

### Step 5: Copy Your Project Files

```bash
# Copy files from your project
cp ~/Downloads/med-image/chest-xray-project/app.py .
cp ~/Downloads/med-image/chest-xray-project/requirements_deployment.txt requirements.txt
cp ~/Downloads/med-image/chest-xray-project/README_SPACE.md README.md

# Copy model
mkdir -p models
cp ~/Downloads/med-image/chest-xray-project/models/best_model.pth models/

# Copy examples
mkdir -p examples
cp ~/Downloads/med-image/chest-xray-project/examples/* examples/ 2>/dev/null || true
```

### Step 6: Commit and Push

```bash
# Add all files
git add .

# Commit
git commit -m "Initial deployment: Chest X-Ray Pneumonia Detection"

# Push to Hugging Face
git push
```

### Step 7: Monitor Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/chest-xray-pneumonia-detector`
2. Click the **"App"** tab
3. Watch build logs
4. App launches automatically when ready!

---

## 📁 Required File Structure

Your Space should have this structure:

```
chest-xray-pneumonia-detector/
├── app.py                    # Main Gradio application
├── requirements.txt          # Python dependencies
├── README.md                 # Space description (shows on page)
├── models/
│   └── best_model.pth       # Your trained model (~100MB)
├── examples/
│   ├── normal1.jpeg         # Example normal X-ray
│   ├── pneumonia1.jpeg      # Example pneumonia X-ray
│   ├── normal2.jpeg
│   └── pneumonia2.jpeg
└── .gitattributes           # Git LFS configuration
```

---

## 🔧 Troubleshooting

### Issue: "Model file too large"
**Solution:** Make sure Git LFS is installed and tracking .pth files
```bash
git lfs install
git lfs track "*.pth"
```

### Issue: "Module not found"
**Solution:** Check requirements.txt has all dependencies:
```txt
gradio==4.8.0
torch==2.0.1
torchvision==0.15.2
Pillow==10.1.0
numpy==1.24.3
```

### Issue: "Out of memory"
**Solution:** 
- Ensure model loads on CPU: `device = torch.device('cpu')`
- Consider model optimization/quantization for very large models

### Issue: "Examples not showing"
**Solution:** 
- Ensure examples/ folder exists with images
- Use relative paths in app.py: `examples/normal1.jpeg`

### Issue: App not starting
**Solution:** 
- Check logs in "App" tab for errors
- Verify all file paths are correct
- Test app locally first: `python app.py`

---

## 🧪 Test Locally Before Deployment

Always test your app locally first:

```bash
cd ~/Downloads/med-image/chest-xray-project

# Test the app
python app.py

# Open browser to: http://localhost:7860
```

If it works locally, it will work on Hugging Face!

---

## ✨ Optional Enhancements

### Add Custom Domain (Free)
Hugging Face provides: `https://huggingface.co/spaces/YOUR_USERNAME/APP_NAME`

### Enable Space Analytics
- Go to Space Settings
- Enable "Analytics" to track usage

### Add More Examples
```bash
# Add more example X-rays to examples/ folder
# Update the Examples section in app.py
```

### Customize Theme
In `app.py`, modify the theme:
```python
with gr.Blocks(theme=gr.themes.Base()) as demo:  # Try: Soft, Base, Glass
```

---

## 📊 Your Live App

Once deployed, your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/chest-xray-pneumonia-detector
```

### Share Your App:
- Direct link: Share the URL
- Embed: Use the embed code from Hugging Face
- Portfolio: Add to your portfolio website

---

## 🎓 For Your Portfolio

**When showcasing this project:**

1. **Link to Space:** Include the live demo link
2. **GitHub Repository:** Host code on GitHub
3. **LinkedIn Post:** Announce your project
4. **Resume:** Add under "Projects" section

**Sample LinkedIn Post:**
```
🚀 Excited to share my latest project: AI-Powered Chest X-Ray Analysis!

Built an end-to-end deep learning system achieving 97.7% accuracy for pneumonia detection.

🔹 Tech: PyTorch, ResNet-50, Gradio
🔹 Deployed on Hugging Face Spaces (FREE!)
🔹 Handled 74% class imbalance professionally
🔹 Comprehensive evaluation with ROC AUC: 0.998

Try it: [Your Space URL]
Code: [Your GitHub URL]

#DataScience #MachineLearning #AI #Healthcare #PyTorch
```

---

## 📞 Need Help?

- **Hugging Face Docs:** https://huggingface.co/docs/hub/spaces
- **Gradio Docs:** https://gradio.app/docs/
- **Community:** https://discuss.huggingface.co/

---

## ✅ Deployment Checklist

Before deploying, ensure:

- [ ] Hugging Face account created
- [ ] Git and Git LFS installed
- [ ] Model file ready (best_model.pth)
- [ ] Example X-ray images prepared
- [ ] App tested locally
- [ ] All file paths are relative (not absolute)
- [ ] README.md completed with your info
- [ ] Requirements.txt is correct
- [ ] Device set to CPU in app.py

---

## 🎉 Congratulations!

Once deployed, you have:
- ✅ A live, interactive AI application
- ✅ Portfolio-ready project
- ✅ Shareable demo for recruiters
- ✅ Experience with ML deployment
- ✅ Free hosting forever!

**Next Steps:**
1. Deploy to Hugging Face Spaces
2. Share on LinkedIn
3. Add to GitHub
4. Include in resume/portfolio
5. Build more projects! 🚀
