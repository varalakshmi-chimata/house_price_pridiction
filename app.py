# pip install flask

from flask import Flask, render_template, request
import numpy as np
import joblib
import os
app = Flask(__name__)

# Load the pre-trained model
path=os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(path,'house_prediction_model.pkl'))

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from user
    bed = int(request.form['bedrooms'])
    bath = int(request.form['bathrooms'])
    loc = int(request.form['location'])
    size = int(request.form['size'])
    status = int(request.form['status'])
    face = int(request.form['facing'])
    Type = int(request.form['type'])

    # Create input data array
    input_data = np.array([[bed, bath, loc, size, status, face, Type]])

    # Predict the price using the pre-trained model
    predicted_price = model.predict(input_data)[0]

    # Render the result page with predicted price
    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run()
