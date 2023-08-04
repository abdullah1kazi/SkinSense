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

## Usage
Instructions on how to use the web application or how to use the model with new data.

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

## Data
The model is trained on a diverse dataset of over 60,000 skin lesion images obtained from the International Skin Imaging Collaboration (ISIC). ISIC is a reputable organization known for curating high-quality dermatology datasets with expertly annotated images. Their dataset includes a comprehensive collection of skin lesion images, covering various skin conditions, both benign and malignant.

## Code
All of our code is open source. In this repository, you'll find everything from the preprocessing steps to the model training code and web application code. Feel free to explore and suggest any improvements.

## Future Improvements
Outline your plans for future updates to the project. 

## Contributing
We welcome contributions! Please see the CONTRIBUTING.md file for details on how you can help improve SkinSense.

## License
This project is licensed under the [license name] - see the LICENSE.md file for details.

## Contact
For questions or feedback, please contact [Your Name](mailto:youremail@example.com).
