from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and encoders
with open('xgmodeling.pkl', 'rb') as model_file:
    xgmodel = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the data from the request
        capital_gain = request.form.get('capital-gain')
        capital_loss = request.form.get('capital-los')
        hours_per_week = request.form.get('hours-per-week')
        education = request.form.get('education_e')
        workclass = request.form.get('workclass_e')
        occupation = request.form.get('occupation_e')
        country = request.form.get('country_e')

        input_data = np.array([capital_gain, capital_loss, hours_per_week, education, workclass, occupation, country])
        input_data_reshaped = input_data.reshape(1, -1)
        # Make the prediction
        output = xgmodel.predict(input_data_reshaped)
        if output[0] == 0:
            A="LESS THAN OR EQUAL TO 50K"
        else:
            A="GREATER THAN 50K"

        # Return the prediction
        return render_template("index.html", prediction_text="YOUR INCOME IS {}".format(A))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)