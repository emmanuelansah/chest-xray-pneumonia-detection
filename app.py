"""
Gradio App for Chest X-Ray Pneumonia Detection

This app provides an interactive interface for pneumonia detection from chest X-rays.
Deploy to Hugging Face Spaces for free hosting.
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
def load_model(model_path='models/best_model.pth'):
    """Load the trained model"""
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model

# Initialize model
try:
    model = load_model()
    model_loaded = True
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model_loaded = False

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Class names
class_names = ['Normal', 'Pneumonia']

def predict(image):
    """
    Predict pneumonia from chest X-ray image
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        dict: Prediction probabilities for each class
        str: Interpretation text with recommendations
    """
    if image is None:
        return None, "Please upload a chest X-ray image."
    
    if not model_loaded:
        return None, "Model not loaded. Please check model file."
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get probabilities for both classes
        probs = probabilities[0].cpu().numpy()
        
        # Create result dictionary for Gradio
        result = {
            class_names[0]: float(probs[0]),
            class_names[1]: float(probs[1])
        }
        
        # Generate interpretation
        prediction = class_names[predicted.item()]
        confidence_score = confidence.item() * 100
        
        if prediction == 'Pneumonia':
            interpretation = f"""
### 🔴 Prediction: **PNEUMONIA DETECTED**
**Confidence: {confidence_score:.2f}%**

#### Analysis:
- The model has detected patterns consistent with pneumonia
- Pneumonia probability: {probs[1]*100:.2f}%
- Normal probability: {probs[0]*100:.2f}%

#### ⚠️ Important Medical Disclaimer:
This is an AI screening tool and **NOT a medical diagnosis**. 

**Required Actions:**
1. ✓ Consult a qualified radiologist or physician immediately
2. ✓ This tool is for educational/screening purposes only
3. ✓ Clinical correlation and additional tests may be needed
4. ✓ Do not make treatment decisions based solely on this result

**Note:** AI models can make errors. Professional medical evaluation is essential.
            """
        else:
            interpretation = f"""
### 🟢 Prediction: **NORMAL**
**Confidence: {confidence_score:.2f}%**

#### Analysis:
- The X-ray appears normal according to the model
- Normal probability: {probs[0]*100:.2f}%
- Pneumonia probability: {probs[1]*100:.2f}%

#### ⚠️ Important Medical Disclaimer:
This is an AI screening tool and **NOT a medical diagnosis**.

**Important Notes:**
1. ✓ A "Normal" result does not rule out all conditions
2. ✓ Consult a healthcare professional for proper evaluation
3. ✓ This tool is for educational/screening purposes only
4. ✓ Regular medical check-ups are recommended

**Note:** AI models can make errors. Professional medical evaluation is essential.
            """
        
        return result, interpretation
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        return None, error_msg

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Chest X-Ray Pneumonia Detection") as demo:
    
    gr.Markdown("""
    # 🏥 Chest X-Ray Pneumonia Detection
    
    Upload a chest X-ray image to detect potential pneumonia using deep learning (ResNet-50).
    
    **⚠️ DISCLAIMER:** This tool is for **educational purposes only** and should NOT be used for actual medical diagnosis. 
    Always consult qualified healthcare professionals for medical advice.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            input_image = gr.Image(
                type="pil", 
                label="Upload Chest X-Ray Image",
                height=400
            )
            
            # Submit button
            submit_btn = gr.Button("🔍 Analyze X-Ray", variant="primary", size="lg")
            
            # Clear button
            clear_btn = gr.ClearButton([input_image], value="Clear", size="sm")
            
            gr.Markdown("""
            ### 📋 Instructions:
            1. Upload a chest X-ray image (JPEG, PNG)
            2. Click "Analyze X-Ray" button
            3. Review the prediction and confidence scores
            4. **Always consult a medical professional**
            
            ### 💡 Tips:
            - Use clear, front-view chest X-rays
            - Ensure good image quality
            - Try the example images below
            """)
        
        with gr.Column(scale=1):
            # Outputs
            output_label = gr.Label(
                label="Prediction Results",
                num_top_classes=2
            )
            
            output_text = gr.Markdown(
                label="Detailed Analysis"
            )
    
    # Examples section
    gr.Markdown("""
    ---
    ### 📸 Try Example X-Rays
    Click on any example below to test the model:
    """)
    
    gr.Examples(
        examples=[
            ["examples/normal1.jpeg"],
            ["examples/pneumonia1.jpeg"],
            ["examples/normal2.jpeg"],
            ["examples/pneumonia2.jpeg"],
        ],
        inputs=input_image,
        label="Sample X-Rays"
    )
    
    # Model information
    gr.Markdown("""
    ---
    ### 📊 Model Information
    
    **Architecture:** ResNet-50 (Transfer Learning)  
    **Training Dataset:** 5,232 chest X-ray images  
    **Test Accuracy:** 97.71%  
    **ROC AUC:** 0.9983  
    
    **Performance Metrics:**
    - Normal Detection: 99.5% recall
    - Pneumonia Detection: 97.1% recall
    - Overall Precision: 97.86%
    
    ### 🛠️ Technical Details
    - **Framework:** PyTorch
    - **Input Size:** 224x224 pixels
    - **Classes:** Normal, Pneumonia
    - **Preprocessing:** ImageNet normalization
    
    ### ⚖️ Limitations
    - Trained on specific dataset (may not generalize to all X-ray types)
    - Cannot detect other lung conditions beyond pneumonia
    - Image quality affects performance
    - Not a replacement for professional medical diagnosis
    
    ### 📚 About This Project
    This is a demonstration project showcasing end-to-end machine learning:
    - Data preprocessing with proper train/val/test splits
    - Transfer learning with ResNet-50
    - Handling class imbalance with weighted loss
    - Comprehensive model evaluation
    - Production-ready deployment
    
    **Created by:** Emmanuel  
    **MSc in Data Science**  
    
    For more projects, visit: [GitHub](#) | [LinkedIn](#) | [Portfolio](#)
    
    ---
    
    ### ⚠️ Final Disclaimer
    **THIS TOOL IS FOR EDUCATIONAL AND DEMONSTRATION PURPOSES ONLY.**
    
    - This is NOT a medical device
    - This is NOT FDA approved
    - Do NOT use for clinical decision making
    - ALWAYS consult qualified healthcare professionals
    - The creators assume NO liability for misuse
    
    If you have medical concerns, please seek immediate professional medical attention.
    """)
    
    # Connect the button to the prediction function
    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[output_label, output_text]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,  # Set to True for temporary public link
        server_name="0.0.0.0",  # Allow external access
        server_port=7860
    )
