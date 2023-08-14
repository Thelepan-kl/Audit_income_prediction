from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and encoders
with open('xgmodeling.pkl', 'rb') as model_file:
    xgmodel = pickle.load(model_file)
with open('encoder1.pkl', 'rb') as encoder1_file:
    loaded_encoder1 = pickle.load(encoder1_file)
with open('encoder2.pkl', 'rb') as encoder2_file:
    loaded_encoder2 = pickle.load(encoder2_file)
with open('encoder3.pkl', 'rb') as encoder3_file:
    loaded_encoder3 = pickle.load(encoder3_file)
with open('encoder4.pkl', 'rb') as encoder4_file:
    loaded_encoder4 = pickle.load(encoder4_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_values = list(request.form.values())
    
    # Convert the first three values to floats
    unencoded_data = [float(x) for x in form_values[:3]]
    
    # Leave the rest of the values as strings
    encoded_data = form_values[3:]
    
    # Perform encoding on the encoded data
    encoded1_input = loaded_encoder1.transform(np.array([encoded_data[0]]).reshape(1, -1))
    encoded2_input = loaded_encoder2.transform(np.array([encoded_data[1]]).reshape(1, -1))
    encoded3_input = loaded_encoder3.transform(np.array([encoded_data[2]]).reshape(1, -1))
    encoded4_input = loaded_encoder4.transform(np.array([encoded_data[3]]).reshape(1, -1))

    final_input = np.append(unencoded_data, encoded1_input)
    final_input = np.append(final_input, encoded2_input)
    final_input = np.append(final_input, encoded3_input)
    final_input = np.append(final_input, encoded4_input)
    
    # Reshape the final input for prediction
    final_input_reshaped = final_input.reshape(1, -1)

    # Concatenate the inputs
    
    # Make prediction using the model
    output = xgmodel.predict(final_input_reshaped)[0]
    if output == 0:
        A="<=50K"
    else:
        A=">50K"
        
    return render_template("index.html", prediction_text="YOUR INCOME IS {}".format(A))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
