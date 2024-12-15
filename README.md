# Alzheimer's detection using ML 
  1. Used OpenCV, Keras, image processing and python concepts.
  2. Created a software which helps to detect the Alzheimer's disease by the help of technology.

#### • Very Mild Demented

<img src="img/26 (45).jpg" width="128"/> <img src="img/26 (46).jpg" width="128"/>

#### • Non-Demented

<img src="img/26 (62).jpg" width="128"/> <img src="img/26 (63).jpg" width="128"/>

#### Program.py

# Alzheimer's Detection Project

## Overview
This project is a **Tkinter-based application** that allows users to classify Alzheimer's disease stages using **Convolutional Neural Networks (CNNs)**. The model is trained on an Alzheimer's dataset with categories such as:

- Mild Demented
- Very Mild Demented
- Moderate Demented
- Non-Demented

The application includes functionalities for model training, result prediction, and an interactive user interface.

## Features
- **Model Training**:
  - Train a CNN model using images from the Alzheimer's dataset.
  - Save the trained model for future predictions.
  
- **Image Selection**:
  - Select an image file for testing and diagnosis.

- **Prediction**:
  - Predict the stage of Alzheimer's disease based on the input image.

- **User Interface**:
  - Intuitive interface using Tkinter.
  - Display training status, file selection, and results seamlessly.

## Requirements
To run this project, the following libraries and tools are required:

### Python Libraries:
- numpy
- matplotlib
- tensorflow
- keras
- tqdm
- opencv-python
- pickle
- tkinter
- PIL (Python Imaging Library)

### Dataset:
- The project uses an **Alzheimer's Dataset** with subdirectories for each class (e.g., MildDemented, VeryMildDemented, etc.). Place the dataset in the following structure:

```
Alzheimer_s Dataset/
├── train/
│   ├── MildDemented/
│   ├── VeryMildDemented/
│   ├── ModerateDemented/
│   ├── NonDemented/
├── test/
    ├── MildDemented/
    ├── VeryMildDemented/
    ├── ModerateDemented/
    ├── NonDemented/
```

### Icons:
Add the following icons to the `icons/` directory:
- `train.png`: Icon for the "Train Model" button.
- `selectfile.png`: Icon for the "Select the Image file to test" button.
- `result.png`: Icon for the "Get Results" button.

## How to Run
1. **Install Dependencies**:
   Use the following command to install all the required Python libraries:
   ```bash
   pip install numpy matplotlib tensorflow keras tqdm opencv-python pillow
   ```

2. **Dataset Setup**:
   Download the Alzheimer's dataset and organize it in the structure mentioned above.

3. **Run the Application**:
   Execute the Python script using the following command:
   ```bash
   python your_script_name.py
   ```

4. **Interact with the UI**:
   - Train the model if a pre-trained model is not available.
   - Select an image file for prediction.
   - Get the classification results in real-time.

## Functionalities
### 1. **Train Model**
- Reads the dataset from the specified directory structure.
- Trains a CNN model using the training and testing data.
- Saves the trained model as `model_pickle.pkl` for future use.

### 2. **Select Image File**
- Allows the user to select an image file for testing.

### 3. **Get Results**
- Predicts the stage of Alzheimer's disease based on the input image.
- Displays the classification result using the pre-trained or newly trained model.

## Key Components
### CNN Model Architecture:
- **Conv2D**: Extracts spatial features from input images.
- **MaxPooling2D**: Reduces spatial dimensions and retains key features.
- **Dropout**: Prevents overfitting.
- **Dense**: Fully connected layers for classification.

### Training Details:
- Input Shape: 45x45x3
- Optimizer: Adadelta
- Loss Function: Categorical Crossentropy
- Epochs: 30
- Batch Size: 128

### File Descriptions:
- `model_pickle.pkl`: Serialized trained model.
- `i_to_l.pkl`: Mapping of class indices to class labels.

## Authors
- **Satyam Gupta**
- **Saurav Sharma**
- **Vaibhav Gupta**
- **Vikash Kumar Singh**

## Acknowledgments
- Special thanks to the creators of the Alzheimer's dataset.
- Guidance and mentorship from faculty and peers.

## Disclaimer
The model is for educational purposes only and not for clinical diagnosis. Unauthorized use of the application is prohibited.


