# Handwritten Digit Classifier

A web application built using **Streamlit** and **TensorFlow** that recognizes handwritten digits (0–9) from images. The model is trained on the popular **MNIST** dataset.


## Features

- Upload a 28x28 grayscale image of a handwritten digit.
- Get instant predictions with confidence scores.
- Built using a Convolutional Neural Network (CNN).
- Simple and interactive Streamlit interface.

---

## Requirements

Install dependencies from the `requirements.txt`:

```bash
pip install -r requirements.txt
Here are the key packages used:

ini
Copy
Edit
absl-py==2.2.2  
altair==5.5.0  
keras==3.10.0  
matplotlib==3.9.4  
numpy==2.0.2  
pandas==2.2.3  
Pillow==11.2.1  
scipy==1.13.1  
streamlit==1.45.1  
tensorflow==2.19.0 

Model Training

The model was trained using the MNIST dataset for 30 epochs with data augmentation to improve performance.

To train the model:

python train_model.py

This will generate a model.h5 file saved in your project directory.

Project Structure

digit_classifier/
│
├── app.py              # Streamlit application
├── train_model.py      # Model training script
├── model.h5            # Saved trained model
├── requirements.txt    # List of dependencies
└── README.md           # This file

Input Format

Accepted file types: JPG, JPEG, PNG

Image size: Automatically resized to 28x28

Color: Converted to grayscale

Tip: Ensure the digit is centered, dark, and drawn clearly to improve accuracy.

Run the Streamlit App

To launch the web application locally:

streamlit run app.py

Then open the provided localhost URL in your browser.

Model Architecture

Conv2D(32) → MaxPooling2D
Conv2D(64) → MaxPooling2D
Flatten → Dense(128) → Dense(64)
Output: Dense(10, softmax)

Optimized using Adam optimizer and categorical crossentropy loss.



Credits

Dataset: MNIST Handwritten Digits

Author: Aleena Varghese

License

This project is open-source and available under the MIT License.