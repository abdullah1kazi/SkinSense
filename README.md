# SkinSense

## Overview
SkinSense is an innovative machine-learning solution designed to diagnose skin diseases using image analysis. We aim to revolutionize how skin diseases are diagnosed by merging the power of machine learning with medical expertise. This tool leverages a sophisticated learning algorithm, ResNet50, optimized with advanced data augmentation techniques to offer a high degree of accuracy and reliability in diagnosing skin diseases.

In its current form, SkinSense is built to diagnose a range of both benign and malignant skin conditions. The model does this by performing two types of classification on the input images. The first is binary classification, differentiating between benign and malignant conditions, while the second is multi-class classification, which identifies specific diagnoses within each category. 

## Model Details
Our model is a sophisticated learning algorithm based on the ResNet50 architecture, a renowned deep learning model for image classification tasks. It's been further refined with a binary and multi-class classifier, enabling it to diagnose both the nature (benign/malignant) and the type of skin disease.

The model is trained using advanced data augmentation techniques that include random rotations, color adjustments, sharpness adjustments, blurring, and image normalization. This not only enriches the diversity of training data but also enables the model to generalize better to new, unseen images.

The training process makes use of K-fold cross-validation to ensure the robustness of the model against overfitting. The learning rate dynamically adjusts over time based on the learning progress, ensuring the model continues to learn and adapt effectively.

The model's performance is impressive, with its ability to differentiate between harmful and non-harmful skin lesions proving crucial in preliminary screenings. Additionally, it performs well in diagnosing specific types of skin diseases, reinforcing its potential as a diagnostic aid for healthcare professionals.

In summary, SkinSense is a comprehensive platform that not only seeks to reduce the time taken to diagnose skin diseases but also improve patient outcomes, ultimately saving lives through early detection and treatment. 

More details about the model architecture, data augmentation techniques, and training process can be found in the code files and documentation provided in this repository.

## Installation
To install the required libraries, use pip and the provided requirements.txt file:

```bash
pip install -r requirements.txt
```
## Data
The model is trained on a diverse dataset of over 60,000 skin lesion images obtained from the International Skin Imaging Collaboration (ISIC). ISIC is a reputable organization known for curating high-quality dermatology datasets with expertly annotated images. Their dataset includes a comprehensive collection of skin lesion images, covering various skin conditions, both benign and malignant.

## Code
All of our code is open source. In this repository, you'll find everything from the preprocessing steps to the model training code and web application code. Feel free to explore and suggest any improvements.

## Usage
Instructions on how to use the web application.

### Running the Web Application Locally

1. **Install Dependencies:**
/```bash
/pip install streamlit torch torchvision pandas base64 numpy
/```

2. **Download the Code and Model:**
- Download the `app.py` file from the GitHub repository and save it to a folder on your local machine.
- Download the pre-trained SkinSense model weights file (`final_model12.pth`) from the GitHub repository and save it in the same folder as `app.py`.

3. **Run the Web Application:**
- Open a terminal or command prompt and navigate to the folder where you saved `app.py` and `final_model12.pth`.
- Execute the following command:
  ```
  streamlit run app.py
  ```

4. **Access the Web Application:**
- After running the above command, Streamlit will start a local server.
- You will see some log messages, and the application will provide you with a local URL (usually http://localhost:8501).
- Open the URL in your web browser to access the SkinSense web application.

### Interacting with the Web Application

1. **Upload an Image:**
- Once you access the web application in your browser, you will see the "Upload your skin lesion image for testing" section.
- Click on the "Browse files" or "Choose File" button to select an image of a skin lesion from your computer. The supported formats are JPG, JPEG, and PNG.

2. **Analyze the Image:**
- After uploading the image, the web application will analyze it using the pre-trained model.
- The application will display the uploaded image along with the predicted diagnosis, which will be either "Benign" or "Malignant," along with the corresponding probability.
- If the model predicts a potential skin condition, additional information about the specific diagnosis (if applicable) will be available in the "More Information" section.  

3. **Interpret the Results:**
- Based on the model's prediction, you will see whether the skin lesion is predicted to be benign or malignant, along with the probability score.
- The application may also provide a specific diagnosis if the model's confidence is above a certain threshold.
- The application also warns that it is for informational purposes only and does not replace professional medical advice.

4. **Test with Different Images:**
- You can upload different skin lesion images to test the model's performance on various cases.
- Observe the model's predictions and the additional information provided in the "More Information" section.

**Remember that this web application is intended for informational purposes only and is not a substitute for professional medical advice.**
If the model predicts a potential skin condition, it's essential to consult a healthcare provider for a definitive diagnosis.

## Future Improvements
We have exciting plans to enhance the SkinSense project further. Here are some of the future updates we are considering:

1. **Data Fine-tuning:** We aim to finetune the model on a more extensive dataset, preferably over millions of diverse skin lesion images. This will allow the model to learn from a broader range of cases and improve its accuracy and generalization.

2. **Web Application:** We are actively working on publishing the SkinSense web application, making it accessible to a broader audience. The website will offer a user-friendly interface, allowing users to upload skin lesion images, change image classification models, receive predictions, and interpret the results seamlessly. Stay tuned for updates as we progress toward the launch of the SkinSense website!\

3. **Mobile Application:** We are actively working on developing a user-friendly mobile application for both iOS and Android platforms. The mobile app will offer seamless access to the SkinSense model, enabling users to analyze skin lesion images conveniently on their smartphones.

4. **Enhanced Model Architecture:** We are continuously exploring advancements in deep learning models and architectures. We plan to integrate the latest state-of-the-art techniques to improve the performance and efficiency of the SkinSense model.

5. **User Feedback and Collaboration:** We value feedback from users and the medical community. By collaborating with dermatologists and medical professionals, we aim to refine the model and make it more reliable and clinically relevant.


We are dedicated to improving and expanding the capabilities of the SkinSense project to benefit both patients and medical practitioners. Your feedback and support are invaluable in our journey to revolutionize skin disease diagnosis. Stay tuned for more updates and follow our GitHub repository for the latest developments.

## Contributing
We welcome contributions! Please see the CONTRIBUTING.md file for details on how you can help improve SkinSense.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact
For questions or feedback, please contact [Abdullah Kazi](kaziabdullah61@gmail.com).
