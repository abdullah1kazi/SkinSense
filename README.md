# SkinSense
Model has not been uploaded, will upload ASAP as .bin and .tar.gz format. 

## Overview
SkinSense is an innovative machine-learning solution I have meticulously designed to revolutionize the diagnosis of skin diseases through advanced image analysis. By seamlessly merging the power of cutting-edge machine learning techniques with medical expertise, SkinSense aims to transform the landscape of dermatological diagnosis. This state-of-the-art tool harnesses the prowess of ResNet50, a sophisticated learning algorithm optimized with advanced data augmentation techniques, to deliver unparalleled accuracy and reliability in identifying a wide array of skin conditions.

In its current form, SkinSense is adept at diagnosing a comprehensive range of both benign and malignant skin diseases. The model achieves this by employing a dual classification approach on the input images. Firstly, it performs binary classification to differentiate between benign and malignant conditions. Secondly, it applies multi-class classification to pinpoint specific diagnoses within each category, providing a granular level of analysis.

## Model Details
At the heart of SkinSense lies a sophisticated learning algorithm built upon the renowned ResNet50 architecture, a deep learning model celebrated for its exceptional performance in image classification tasks. I have further refined this model by incorporating a binary and multi-class classifier, enabling it to diagnose both the nature (benign/malignant) and the specific type of skin disease with remarkable precision.

To ensure the model's robustness and generalization capabilities, I have employed advanced data augmentation techniques during the training process. These techniques include random rotations, color adjustments, sharpness adjustments, blurring, and image normalization. By enriching the diversity of the training data, the model gains the ability to adapt and perform exceptionally well on new, unseen images.

To mitigate the risk of overfitting, I have implemented K-fold cross-validation throughout the training process. This approach guarantees the model's stability and reliability. Additionally, the learning rate dynamically adjusts over time based on the learning progress, allowing the model to continuously learn and adapt effectively.

The performance of SkinSense is truly remarkable. Its ability to differentiate between harmful and non-harmful skin lesions proves invaluable in preliminary screenings. Moreover, it excels in diagnosing specific types of skin diseases, solidifying its potential as a powerful diagnostic aid for healthcare professionals.

In essence, SkinSense represents a comprehensive platform that not only expedites the diagnosis of skin diseases but also holds the promise of improving patient outcomes and ultimately saving lives through early detection and timely treatment.

For a deeper dive into the model architecture, data augmentation techniques, and training process, I encourage you to explore the code files and documentation provided in this repository.

## Installation
To install the required libraries, simply use pip and the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

## Data
SkinSense is trained on a rich and diverse dataset comprising over 60,000 skin lesion images sourced from the esteemed International Skin Imaging Collaboration (ISIC). ISIC is a reputable organization renowned for curating high-quality dermatology datasets with expertly annotated images. Their dataset encompasses a comprehensive collection of skin lesion images, spanning various skin conditions, both benign and malignant.

## Code
In the spirit of transparency and collaboration, all of the code for SkinSense is open source. Within this repository, you will find everything from the preprocessing steps to the model training code and web application code. I wholeheartedly invite you to explore, scrutinize, and suggest any improvements that can further enhance the project.

## Usage
Below are the instructions on how to use the SkinSense web application.

### Running the Web Application Locally

1. **Install Dependencies:**

```bash
pip install streamlit torch torchvision pandas base64 numpy
```

2. **Download the Code and Model:**
- Download the `app.py` file from the GitHub repository and save it to a folder on your local machine.
- Download the pre-trained SkinSense model weights file (`pytorch_model.bin`) and the `config.json` file from the [HuggingFace repository](https://huggingface.co/Akazi/Resnet101FinetunedModelSkinSense) and save them in the same folder as `app.py`.

3. **Run the Web Application:**
- Open a terminal or command prompt and navigate to the folder where you saved `app.py` and `pytorch_model.bin`.
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
- The application will display the uploaded image along with the predicted diagnosis, which will be either "Benign" or "Malignant," accompanied by the corresponding probability.
- If the model predicts a potential skin condition, additional information about the specific diagnosis (if applicable) will be available in the "More Information" section.

3. **Interpret the Results:**
- Based on the model's prediction, you will see whether the skin lesion is predicted to be benign or malignant, along with the probability score.
- The application may also provide a specific diagnosis if the model's confidence exceeds a certain threshold.
- It is important to note that the application is intended for informational purposes only and does not replace professional medical advice.

4. **Test with Different Images:**
- You can upload various skin lesion images to evaluate the model's performance across a range of cases.
- Observe the model's predictions and the additional information provided in the "More Information" section.

Please remember that this web application is designed for informational purposes and should not be considered a substitute for professional medical advice. If the model predicts a potential skin condition, it is crucial to consult a healthcare provider for a definitive diagnosis.

## Future Improvements
I have an exciting roadmap to further enhance the SkinSense project. Here are some of the future updates I am actively working on:

1. **Data Fine-tuning:** My aim is to fine-tune the model on an even more extensive dataset, ideally encompassing millions of diverse skin lesion images. By exposing the model to a broader range of cases, I can further improve its accuracy and generalization capabilities.

2. **Web Application:** I am diligently working on publishing the SkinSense web application, making it accessible to a wider audience. The website will offer a user-friendly interface, allowing users to effortlessly upload skin lesion images, change image classification models, receive predictions, and interpret the results seamlessly. Stay tuned for updates as I progress towards the launch of the SkinSense website!

3. **Mobile Application:** Recognizing the importance of accessibility, I am actively developing a user-friendly mobile application for both iOS and Android platforms. The mobile app will provide seamless access to the SkinSense model, enabling users to conveniently analyze skin lesion images directly from their smartphones.

4. **Enhanced Model Architecture:** I am continuously exploring advancements in deep learning models and architectures. By integrating the latest state-of-the-art techniques, I aim to further improve the performance and efficiency of the SkinSense model.

5. **User Feedback and Collaboration:** I greatly value feedback from users and the medical community. By collaborating with dermatologists and medical professionals, I strive to refine the model and ensure its reliability and clinical relevance.

I am deeply committed to improving and expanding the capabilities of the SkinSense project to benefit both patients and medical practitioners. Your feedback and support are invaluable in my journey to revolutionize skin disease diagnosis. Stay tuned for more updates and follow my GitHub repository for the latest developments.

## Contributing
I warmly welcome contributions from the community! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines on how you can help improve SkinSense.

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CONTRIBUTING.md)

## License
This project is licensed under the MIT License. For more information, please see the [LICENSE.md](LICENSE.md) file.

## Contact
If you have any questions, feedback, or inquiries, please feel free to reach out to me, Abdullah Kazi, at [kaziabdullah61@gmail.com](mailto:kaziabdullah61@gmail.com). I am always happy to engage in meaningful discussions and collaborate on improving SkinSense.
