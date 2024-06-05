from flask import Flask, render_template, request
from image_processor import ImageProcessor
from prediction import Prediction

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Get the image and prediction number from the form
    image = request.form.get('image')
    prediction_number = int(request.form.get('prediction_number'))
    

    # Preprocess the image
    image_processor = ImageProcessor()
    image_processed = image_processor.preprocess_images(image)

    # Make prediction with the specified number
    prediction_model = Prediction(train_model="/workspaces/Face-Aging/trainned_model/test.h5")
    new_image = prediction_model.predict(image_processed, prediction_number)

    return render_template('result.html', script=new_image)

if __name__ == "__main__":
    app.run(debug=True)
