# Import necessary libraries
from flask import Flask, jsonify, make_response, redirect, render_template, request, url_for
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model, Model
from io import BytesIO
from matplotlib import cm
from PIL import Image
import base64
import cv2
import numpy as np
import os
import pickle
import random
import xgboost as xgb

# Define Flask app and configure environment variables
app = Flask(__name__)
app.config.from_pyfile('config.py')

# Define path to models directory
models_dir = 'models'

# Load trained CNN model for feature extraction
cnn_model = load_model(os.path.join(models_dir, 'VGG16-Model.h5'))

# Load trained XGBoost model for image classification
xgb_model = pickle.load(open(os.path.join(models_dir, 'VGG16-XGBoost-Final-Tuned-Hybrid-Model.pkl'), 'rb'))

# Load chest X-ray verifier model
chest_xray_verifier_model = load_model(os.path.join(models_dir, 'VGG16-Chest-X-Ray-Verifier-Model.h5'))

# Define home route for GET and POST requests
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return render_template('main.html', error='Invalid Method')  # If POST request, render main template with error message
    return render_template('main.html')  # If GET request, simply render main template

# Define function to preprocess input images for optimal deep learning performance
def preprocess_image(image):
    image = cv2.resize(image, (256, 256))  # Resize image to standard size of 256x256 pixels
    image = image / 255.0  # Normalize pixel values to range [0, 1]
    return image

# Define function to verify whether input image is chest X-ray
def verify_chest_xray(image_tensor):
    prediction = chest_xray_verifier_model.predict(image_tensor)  # Predict with chest X-ray verifier model
    chest_xray_prob = 1 - prediction[0][0]  # Calculate probability of image being chest X-ray
    return chest_xray_prob

# Define function to generate Grad-CAM visualization for given image and CNN model
def grad_cam(model, image, layer_name):

    # Define model that generates both final model predictions as well as output of chosen layer
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    # Incoming image is singular example so expand dimensions to represent batch of size 1
    image_tensor = np.expand_dims(image, axis=0)

    # Cast image tensor to float32 type
    inputs = tf.cast(image_tensor, tf.float32)

    # Set up gradient tape to monitor intermediate variables and predictions
    with tf.GradientTape() as tape:

        # Extract activations from chosen layer and model's final predictions
        last_conv_layer_output, preds = grad_model(inputs)

        # Identify predicted class from final predictions
        pred_class = tf.argmax(preds[0])

        # Get output of predicted class from final layer
        class_channel = preds[:, pred_class]

    # Compute gradient of output with respect to chosen layer's output
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Reduce 2D gradients to 1D by averaging across height and width dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply 2D output map of chosen layer by 1D pooled gradients
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap to be between 0 and 1 for better visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Return Grad-CAM heatmap
    return heatmap.numpy()

# Define function to convert image to Base64 for transmission to client
def image_to_base64(image):
    image_pil = Image.fromarray(image)  # Convert image array to PIL image
    buff = BytesIO()  # Initialize bytes buffer
    image_pil.save(buff, format='PNG')  # Save image to buffer in PNG format
    image_base64 = base64.b64encode(buff.getvalue()).decode('utf-8')  # Convert buffer contents to Base64 and decode to string
    return image_base64

# Define route for prediction, only accept POST requests
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Check if request contains file
        if 'file' not in request.files:
            return jsonify(error='Bad Request'), 400

        # Get file from request
        file = request.files['file']

        # Check if file name is empty
        if file.filename == '':
            return jsonify(error='Bad Request'), 400

        # Load file data into numpy array
        filestr = file.read()
        np_image = np.fromstring(filestr, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Check if image is empty
        if image is None or len(image) == 0:
            return jsonify(error='Bad Request'), 400

        # Preprocess image for model
        image = preprocess_image(image)

        # Convert image to tensor by adding batch dimension
        image_tensor = np.expand_dims(image, axis=0)

        # Verify whether image is chest X-ray
        chest_xray_probability = verify_chest_xray(image_tensor)
        chest_xray_probability = f'{chest_xray_probability * 100:.1f}'

        # Pass image through CNN model to get features
        cnn_features = cnn_model.predict(image_tensor)

        # Flatten CNN features to pass to XGBoost model
        cnn_features = cnn_features.flatten()

        # Use XGBoost model to predict class of image
        prediction_probabilities = xgb_model.predict_proba(np.expand_dims(cnn_features, axis=0))

        # Find predicted class index
        prediction_index = np.argmax(prediction_probabilities[0])

        # Map prediction index to class label
        prediction_class = ['COVID-19', 'Pneumonia', 'Normal'][prediction_index]

        # Get probability of predicted class
        prediction_probability = prediction_probabilities[0][prediction_index]
        prediction_probability = f'{prediction_probability * 100:.1f}'

        # Generate Grad-CAM heatmap for visualization
        grad_cam_image = grad_cam(cnn_model, image, 'block5_conv3')

        # Enhance heatmap image for better visualization
        grad_cam_image = np.maximum(grad_cam_image, 0)
        grad_cam_image = np.minimum(grad_cam_image, 1)
        heatmap_colored = cm.jet(grad_cam_image)[:, :, :3]
        heatmap_colored = np.uint8(255 * heatmap_colored)

        # Resize heatmap to original image size
        heatmap_resized = np.array(Image.fromarray(heatmap_colored).resize((image.shape[1], image.shape[0])))

        # Convert image and heatmap to 0-255 scale
        if image.max() <= 1:
            image = (image * 255).astype('uint8')
        if heatmap_resized.max() <= 1:
            heatmap_resized = (heatmap_resized * 255).astype('uint8')

        # Superimpose heatmap on original image, with more weight on original image
        superimposed_image = heatmap_resized * 0.4 + image * 0.6
        superimposed_image = np.clip(superimposed_image, 0, 255).astype('uint8')

        # Convert images to Base64 for transmission to client
        original_image_base64 = image_to_base64(image)
        heatmap_base64 = image_to_base64(heatmap_resized)
        superimposed_image_base64 = image_to_base64(superimposed_image)

        # Return prediction and images to client
        return jsonify({
            'prediction_class': prediction_class,
            'prediction_probability': prediction_probability,
            'chest_xray_probability': chest_xray_probability,
            'original_image': 'data:image/png;base64,' + original_image_base64,
            'heatmap_image': 'data:image/png;base64,' + heatmap_base64,
            'superimposed_image': 'data:image/png;base64,' + superimposed_image_base64,
        })

    except Exception as e:

        # Return internal server error for any exception during processing
        return jsonify(error='Internal Server Error'), 500

# Define route for fetching random example image, only accept POST requests
@app.route('/random-example', methods=['POST'])
def random_example():

    # Define path to example images directory
    images_dir = os.path.join('static', 'examples')

    # List all image files in directory
    images = os.listdir(images_dir)

    # Choose random image file
    random_image = random.choice(images)

    # Create full path to image file
    image_path = os.path.join(images_dir, random_image)

    # Open image file in binary mode
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()  # Read image file into memory

    # Create response with image data
    response = make_response(image_bytes)

    # Set headers for response to indicate it is PNG image
    response.headers.set('Content-Type', 'image/png')
    response.headers.set('Content-Disposition', 'attachment', filename=f'{random_image}')

    # Return image data to client
    return response

# Define error handler for 404 errors
@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('home'))  # Redirect to home route for 404 errors

# Define error handler for 405 errors
@app.errorhandler(405)
def method_not_allowed(e):
    if request.path == '/predict' or request.path == '/random-example':
        return redirect(url_for('home'))  # If 405 error occurs at '/predict' or '/random-example', redirect to home route
    else:
        return 'Method Not Allowed', 405  # Otherwise, return default 405 response

# Start Flask app
if __name__ == '__main__':
    app.run(debug=False, port=os.getenv('PORT', 5000))