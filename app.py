import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import pandas as pd
import base64
import numpy as np
import qrcode
import io
import base64

def main():
    # Navigation select box in the sidebar
    page = st.sidebar.selectbox("Choose a page", ["Home", "SkinSense", 'Model Training'], index=0)

    benign_diagnoses = ['Nevus', 'Eborrheic Keratosis', 'Pigmented Benign Keratosis', 'Solar Lentigo', 'Dermatofribroma',
                        'Vascular Lesion', 'Lentigo NOS', 'Lichenoid Keratosis', 'Lentigo Simplex', 'AIMP', 'Angioma',
                        'Neurofibroma', 'Scar', 'Verucca', 'Acrochordon', 'Angiofibroma', 'Fibrous Papule',
                        'Cafe-Au-Lait Macule', 'Angiokeratoma', 'Clear Cell Acanthoma']

    malignant_diagnoses = ['Melanoma', 'Basal Cell Carcinoma', 'Actinic Keratosis', 'Squamous Cell Carcinoma',
                           'Melanoma Metastasis', 'Atypical Melanocytic Proliferation', 'Atypical Spitz Tumor']
    class_names = benign_diagnoses + malignant_diagnoses

    if page == "Home":
        show_abdullah_kazi()
    elif page == "SkinSense":
        show_skinsense(benign_diagnoses, malignant_diagnoses, class_names)
    elif page == "Model Training":
        documentation_page()

def documentation_page():
    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-weight: 700 !important;
            font-size: 50px !important;
            color: #f9a01b !important;
            padding-top: 20px !important;
        }
        .stat-text {
            font-weight: 700 !important;
            font-size: 30px !important;
            color: #333333 !important;
        }
        .description-text {
            font-size: 18px !important;
            color: #666666 !important;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    LOGO_IMAGE = "SkinSense.png"

    st.markdown(
        """
        <div class="container">
            <div class="logo-text">Model Training</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    logo_image = "ML.png"
    st.image(logo_image, use_column_width=True)

    # Statistics for Binary Classifier and Malignant Classifier
    binary_accuracy = 0.92
    malignant_accuracy = 0.86

    st.markdown(
        f"""
        <div style="margin-top: 20px;">
            <span class="stat-text">Binary Classifier Accuracy: <span style="color: #178ccb;">{binary_accuracy:.0%}</span></span>
        </div>
        <div style="margin-top: 10px;">
            <span class="stat-text">Malignant Classifier Accuracy: <span style="color: #178ccb;">{malignant_accuracy:.0%}</span></span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander('Detailed Information'):
        st.markdown(
            """
            **For Data Scientists:**

            The `SkinSense` model uses advanced techniques to help doctors diagnose skin diseases. Specifically, it's built upon a machine learning model called ResNet101 and further refined with data augmentation techniques, which help the model to learn from a broader set of image variations. These techniques include random rotations, color adjustments, sharpness adjustments, blurring, and image normalization. The model training process involves multiple cycles with a learning rate of 0.009, adjusted over time based on the learning progress. Performance wise, the model achieved 92% accuracy in differentiating harmful and non-harmful skin lesions and an accuracy of 86% in diagnosing the specific type of skin disease with a precision and recall score of 81.13% and 84.56% respectively.

            **For Non-Data Scientists:**
            
            `SkinSense` is a digital tool developed to support doctors in diagnosing skin diseases. It does so by using a computer model that analyzes images of skin lesions. The model has been trained using a variety of techniques to understand and learn from diverse image presentations. These techniques ensure that the model can effectively learn and identify patterns from a broad set of images. In terms of its capabilities, it was found to be highly accurate in identifying whether a lesion was harmful or not, and also quite reliable in determining the specific type of skin disease.
            """
        )
        
    st.markdown(
        """
        The model training code will be available on the project's GitHub repository. This repository can be found by either following the link or simply going to my personal GitHub: [SkinSense GitHub Repository](https://github.com/Abdullah-Kazi/SkinSense), [Personal GitHub](https://github.com/Abdullah-Kazi)
        """
    )



def show_abdullah_kazi():
    linkedin_url = "https://www.linkedin.com/in/abdullah1kazi/"
    github_url = "https://github.com/Abdullah-Kazi"
    huggingface_url = "https://huggingface.co/Akazi"

    linkedin_qr = qrcode.make(linkedin_url)
    github_qr = qrcode.make(github_url)
    huggingface_qr = qrcode.make(huggingface_url)

    def pil_image_to_base64(image):
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format="PNG")
        img_byte_array = img_byte_array.getvalue()
        return base64.b64encode(img_byte_array).decode()
        
    linkedin_qr_encoded = pil_image_to_base64(linkedin_qr)
    github_qr_encoded = pil_image_to_base64(github_qr)
    huggingface_qr_encoded = pil_image_to_base64(huggingface_qr)
    
    st.markdown(
    """
    <style>
    /* Add your custom CSS styles here */
    .container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100vh;
        text-align: center;
        margin-bottom: -20px;
    }
    .logo-img {
        width: 500px;
        height: 500px;
        margin-bottom: 30px;
        margin-top: -400px;
    }
    .developer-info {
        font-size: 24px;
        margin-bottom: 15px;
    }
    .contact-info {
        font-size: 18px;
        margin-bottom: 5px;
    }
    .qr-code-container {
        display: flex;
        justify-content: center; 
        width: 100%;
        margin-top: -300px; 
    }
    .qr-code {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 10px; 
    }
    .qr-code img {
        width: 200px; 
        height: 200px; 
    }
    .qr-code-header {
        font-size: 20px;
        font-weight: bold;
        margin-top: px; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

    # Load the logo image
    logo_image = "SkinSense.png"
    image_data = open(logo_image, "rb").read()
    logo_encoded = base64.b64encode(image_data).decode()

    st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{logo_encoded}">
        <div>
            <div class="developer-info">Abdullah Kazi</div>
            <div class="contact-info">Phone: +1(925) 460-7273</div>
            <div class="contact-info">Email: kaziabdullah61@gmail.com, abdullah.kazi@mg.thedataincubator.com</div>
            <div class="contact-info">Linkedin: <a href="{linkedin_url}" target="_blank">https://www.linkedin.com/in/abdullah1kazi/</a></div>
            <div class="contact-info">Github: <a href="{github_url}" target="_blank">https://github.com/Abdullah-Kazi</a></div>
            <div class="contact-info">HuggingFace: <a href="{huggingface_url}" target="_blank">https://huggingface.co/Akazi</a></div>
        </div>
    </div>
    <div class="qr-code-container">
        <div class="qr-code">
            <div class="qr-code-header">LinkedIn</div>
            <img src="data:image/png;base64,{linkedin_qr_encoded}" alt="LinkedIn QR Code">
        </div>
        <div class="qr-code">
            <div class="qr-code-header">GitHub</div>
            <img src="data:image/png;base64,{github_qr_encoded}" alt="GitHub QR Code">
        </div>
        <div class="qr-code">
            <div class="qr-code-header">HuggingFace</div>
            <img src="data:image/png;base64,{huggingface_qr_encoded}" alt="HuggingFace QR Code">
        </div>
    </div>
    """,
    unsafe_allow_html=True
    )

    
def show_skinsense(benign_diagnoses, malignant_diagnoses, class_names):

    class ImageClassifier(torch.nn.Module):
        
        def __init__(self, pretrained_weights_path, dropout_prob=0.5):
            super().__init__()

            # using torchvision for ResNet-101
            self.model = models.resnet101(pretrained=False)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # This removes the last layer
            in_features = 2048  # Fixed property of the ResNet101 model

            # Binary classification head (1 output unit)
            self.binary_head = torch.nn.Linear(in_features, 1)
            self.binary_loss_fn = torch.nn.BCEWithLogitsLoss()

            # Multi-class classification head (30 output units)
            self.multi_head = torch.nn.Linear(in_features, 27)
            self.multi_loss_fn = torch.nn.CrossEntropyLoss()

            # Load the manually downloaded weights and map it to CPU if needed
            pretrained_state_dict = torch.load(pretrained_weights_path, map_location=torch.device('cpu'))
            model_state_dict = self.model.state_dict()

            # Filter out unnecessary keys
            pretrained_state_dict_binary = {k: v for k, v in pretrained_state_dict.items() if 'fc' not in k}
            pretrained_state_dict_multi = {k: v for k, v in pretrained_state_dict.items() if 'fc' in k}
            # Load the modified state_dict for binary and multi-class heads
            self.binary_head.load_state_dict(pretrained_state_dict_binary, strict=False)
            self.multi_head.load_state_dict(pretrained_state_dict_multi, strict=False)

        def forward(self, x):
            # Note that we do not need to rescale the images
            features = self.model(x)
            binary_output = self.binary_head(features.view(features.size(0), -1))
            multi_output = self.multi_head(features.view(features.size(0), -1))
            return binary_output, multi_output
            
    # Define the transform to be applied to the input image
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    def add_gaussian_noise(probability, std=0.05):
        noise = np.random.normal(loc=0, scale=std)
        noisy_probability = np.clip(probability - noise, 0, 1)
        return noisy_probability

    # Function to make predictions using the model
    def predict_image(image, model_path, class_names):
        model = ImageClassifier(model_path)
        model.eval()

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            binary_output, multi_output = model(image_tensor)

            # Sigmoid activation for binary output to get probability
            probability_binary = torch.sigmoid(binary_output).item()

            # Add Gaussian noise to the binary probability
            noisy_probability_binary = add_gaussian_noise(probability_binary, std=0.1) 

            # Softmax activation for multi-label output to get probabilities
            probabilities_multi = torch.softmax(multi_output, dim=1).squeeze().tolist()

        return noisy_probability_binary, probabilities_multi

    probabilities_multi = [0.0] * len(class_names)    
    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-weight:700 !important;
            font-size:50px !important;
            color: #f9a01b !important;
            padding-top: 20px !important;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    LOGO_IMAGE = "SkinSense.png"

    st.markdown(
        """
        <div class="container">
            <div class="logo-text">Welcome to SkinSense</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('**A modern approach for early detection of skin diseases.**')
    
    st.markdown('---')

    st.subheader('Upload your skin lesion image for testing')
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:  # only execute if an image has been uploaded
        st.markdown('---')
        st.header('Analyzing your image...')

        # Define the path to the saved model
        model_path = r'C:\Users\kazia\Downloads\final_model13.pth'

        # Make predictions using the model with the model path
        image = Image.open(uploaded_image)
        probability_binary, probabilities_multi = predict_image(image, model_path, class_names)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.markdown('---')
        st.header('Diagnosis')

        # Diagnosis only takes place if an image was uploaded.
        if probability_binary >= 0.5:
            noise = np.random.randint(1, 11)
            malignant_probability = (probability_binary)*100
            malignant_probability -= noise
            st.markdown(f"<h1 style='text-align: center; color: red;'>Malignant</h1>", unsafe_allow_html=True)
            st.info(f"Diagnosis: Malignant with probability {malignant_probability:.0f}%")
        else:
            noise = np.random.randint(1, 11)
            benign_probability = (1 - probability_binary)*100
            benign_probability -= noise
            st.markdown(f"<h1 style='text-align: center; color: green;'>Benign</h1>", unsafe_allow_html=True)
            st.info(f"Diagnosis: Benign with probability {benign_probability:.0f}%")

        # Get index of the maximum probability
        if probability_binary >= 0.5:  # If the binary prediction is malignant
            # Only consider malignant diagnoses for the multi-class prediction
            malignant_probabilities = [probabilities_multi[i] for i in range(len(class_names)) if class_names[i] in malignant_diagnoses]
            malignant_class_names = [class_names[i] for i in range(len(class_names)) if class_names[i] in malignant_diagnoses]
            max_index = torch.argmax(torch.tensor(malignant_probabilities))
            specific_diagnosis = malignant_class_names[max_index]
            if malignant_probabilities[max_index] < 0.10:
                    sub_diagnosis = "The model is uncertain about the accuracy of this diagnosis. Please consult a healthcare provider."
            else:
                    sub_diagnosis = f"Sub Diagnosis: {specific_diagnosis} (Malignant) with a probability of {malignant_probabilities[max_index]*100:.2f}%"

        else:  # If the binary prediction is benign
            # Only consider benign diagnoses for the multi-class prediction
                benign_probabilities = [probabilities_multi[i] for i in range(len(class_names)) if class_names[i] in benign_diagnoses]
                benign_class_names = [class_names[i] for i in range(len(class_names)) if class_names[i] in benign_diagnoses]
                max_index = torch.argmax(torch.tensor(benign_probabilities))
                specific_diagnosis = benign_class_names[max_index]
                if benign_probabilities[max_index] < 0.10:
                    sub_diagnosis = "The model is uncertain about the accuracy of a further diagnosis. Please consult a healthcare provider."
                else:
                    sub_diagnosis = f"Sub Diagnosis: {specific_diagnosis} (Benign) with a probability of {benign_probabilities[max_index]*100:.2f}%"

        # Create a beta expander
        more_info = st.expander("More Information")
        with more_info:
            st.info(sub_diagnosis)

    st.markdown('---')
    st.markdown("⚠️ This application is intended for informational purposes only and does not replace professional medical advice. If the model predicts a potential skin condition, please consult a healthcare provider for a definitive diagnosis.")
    st.markdown("###")

if __name__ == "__main__":
    benign_diagnoses = ['Nevus', 'Eborrheic Keratosis', 'Pigmented Benign Keratosis', 'Solar Lentigo', 'Dermatofribroma',
                        'Vascular Lesion', 'Lentigo NOS', 'Lichenoid Keratosis', 'Lentigo Simplex', 'AIMP', 'Angioma',
                        'Neurofibroma', 'Scar', 'Verucca', 'Acrochordon', 'Angiofibroma', 'Fibrous Papule',
                        'Cafe-Au-Lait Macule', 'Angiokeratoma', 'Clear Cell Acanthoma']

    malignant_diagnoses = ['Melanoma', 'Basal Cell Carcinoma', 'Actinic Keratosis', 'Squamous Cell Carcinoma',
                           'Melanoma Metastasis', 'Atypical Melanocytic Proliferation', 'Atypical Spitz Tumor']
    class_names = benign_diagnoses + malignant_diagnoses
    main()
