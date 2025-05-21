# Handwritten Digit Classifier

A machine learning project that recognizes handwritten digits (0–9) using TensorFlow and Keras. It includes a Streamlit web app to upload images and get real-time digit predictions.

## Features
- Trains a neural network on MNIST dataset
- Multi-class classification (digits 0–9)
- Streamlit app for easy digit prediction
- Model trained for 30 epochs with normalization and augmentation
- Saved model reusable for predictions

## Technologies
Python 3.9+, TensorFlow, Keras, Streamlit, NumPy, Pillow, SciPy

## Installation & Usage
1. Clone repo:
   ```bash
   git clone https://github.com/aleenavarghese29/digit-classifier.git

   cd digit-classifier
````

2. (Optional) Create & activate virtual env:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Train model:

   ```bash
   python train_model.py
   ```
5. Run Streamlit app:

   ```bash
   streamlit run app.py
   ```
6. Open browser at `http://localhost:8501` and upload 28x28 grayscale images for predictions.

## Project Files

* `train_model.py`: Model training script
* `app.py`: Streamlit app
* `model.h5`: Saved model
* `requirements.txt`: Dependencies
* `.gitignore`: Files ignored by git

## License

MIT License

