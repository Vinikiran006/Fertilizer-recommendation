# from flask import Flask, request, render_template
# import joblib
# import pandas as pd
# import os

# # Initialize the Flask application
# app = Flask(__name__)

# # --- Model Loading ---
# # Construct the full path to the model file
# # This makes the app more robust, especially when deploying
# model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fertilizer_pipeline.joblib')

# # Load the trained machine learning pipeline
# try:
#     model_pipeline = joblib.load(model_path)
#     print("Model loaded successfully.")
# except FileNotFoundError:
#     print(f"Error: Model file not found at {model_path}. Please run the main.py script to generate it.")
#     model_pipeline = None
# except Exception as e:
#     print(f"An error occurred while loading the model: {e}")
#     model_pipeline = None


# # --- Web Routes ---

# # Define the route for the home page, which displays the form
# @app.route('/')
# def home():
#     # Renders the HTML form template
#     return render_template('index.html')

# # Define the route that handles the form submission and makes a prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     if model_pipeline is None:
#         return render_template('index.html', prediction_text='Error: Model is not loaded. Please check the server logs.')

#     try:
#         # Get the data from the form
#         input_data = {
#             'Temperature': [float(request.form['temperature'])],
#             'Humidity': [float(request.form['humidity'])],
#             'Soil Moisture': [float(request.form['soil_moisture'])],
#             'Soil Type': [request.form['soil_type']],
#             'Crop Type': [request.form['crop_type']],
#             'Nitrogen': [float(request.form['nitrogen'])],
#             'Potassium': [float(request.form['potassium'])],
#             'Phosphorus': [float(request.form['phosphorus'])]
#         }

#         # Convert the dictionary to a pandas DataFrame, as the model expects it
#         input_df = pd.DataFrame(input_data)
#         print(f"Received input data for prediction: \n{input_df}")

#         # Make prediction using the loaded pipeline
#         prediction = model_pipeline.predict(input_df)[0]
#         print(f"Prediction result: {prediction}")

#         # Pass the prediction result back to the HTML template
#         return render_template('index.html', prediction_text=f'Recommended Fertilizer: {prediction}')

#     except Exception as e:
#         print(f"An error occurred during prediction: {e}")
#         # In case of an error, show an informative message to the user
#         return render_template('index.html', prediction_text=f'An error occurred. Please check the input values. Details: {e}')


# # This is required to run the app
# if __name__ == '__main__':
#     # The debug=True flag allows you to see errors in the browser and auto-reloads the server on code changes
#     app.run(debug=True)




from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Model Loading ---
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fertilizer_pipeline.joblib')
try:
    model_pipeline = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model_pipeline = None

# --- Fertilizer Information Database ---
# A dictionary to hold descriptions and usage advice for each fertilizer
fertilizer_info = {
    'Urea': {
        'description': 'Urea is a high-nitrogen fertilizer (46% N). It is crucial for promoting lush, green leafy growth and is a primary component of photosynthesis.',
        'usage': 'Best applied during the main growth period (vegetative stage). For most crops, this is 2-4 weeks after planting. It can be applied as a top dressing.'
    },
    'DAP': {
        'description': 'Di-Ammonium Phosphate (DAP) is rich in Phosphorus (46% P) and contains Nitrogen (18% N). It is vital for strong root development and energy transfer.',
        'usage': 'Ideal as a basal fertilizer. Apply and mix into the soil at the time of sowing or planting to ensure seedlings develop a robust root system.'
    },
    'MOP': {
        'description': 'Muriate of Potash (Potassium Chloride) is a high-potassium fertilizer (60% K). It improves overall plant health, disease resistance, and the quality of fruits and flowers.',
        'usage': 'Apply during the later stages of growth, particularly during flowering and fruit development, to improve yield and quality.'
    },
    'Organic compost': {
        'description': 'A natural soil conditioner made from decomposed organic matter. It improves soil structure, water retention, and provides a slow, balanced release of essential nutrients.',
        'usage': 'Best incorporated into the soil before planting. It can also be used as a mulch or top dressing for existing plants throughout the season.'
    },
    'SSP': {
        'description': 'Single Super Phosphate (SSP) provides Phosphorus (16% P), Calcium, and Sulphur. It is an excellent choice for promoting root growth and is particularly beneficial for oilseeds and pulses.',
        'usage': 'Apply as a basal dose at the time of planting. It is water-soluble and quickly becomes available to young plants.'
    },
    # Default for generic NPK compound fertilizers
    'default': {
        'description': 'This is a compound NPK fertilizer, providing a balanced mix of Nitrogen (N), Phosphorus (P), and Potassium (K) in the specified ratio.',
        'usage': 'Compound fertilizers are versatile. They are often used as a foundational (basal) dose at planting, but timing can vary based on the specific N-P-K ratio and the crop\'s needs.'
    }
}
# Add all NPK compound names to the dictionary, pointing them to the 'default' info
for n in range(0, 121, 5):
    for p in range(0, 61, 5):
        for k in range(0, 61, 5):
            key = f"{n}-{p}-{k}"
            if key not in fertilizer_info:
                fertilizer_info[key] = fertilizer_info['default']


# --- Web Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return render_template('index.html', error='Model is not loaded. Please check server logs.')

    try:
        input_data = {
            'Temperature': [float(request.form['temperature'])],
            'Humidity': [float(request.form['humidity'])],
            'Soil Moisture': [float(request.form['soil_moisture'])],
            'Soil Type': [request.form['soil_type']],
            'Crop Type': [request.form['crop_type']],
            'Nitrogen': [float(request.form['nitrogen'])],
            'Potassium': [float(request.form['potassium'])],
            'Phosphorus': [float(request.form['phosphorus'])]
        }
        input_df = pd.DataFrame(input_data)
        
        prediction = model_pipeline.predict(input_df)[0]
        
        # Get the info for the predicted fertilizer
        info = fertilizer_info.get(prediction, fertilizer_info['default'])

        # Pass a dictionary with all the info to the template
        prediction_result = {
            'name': prediction,
            'description': info['description'],
            'usage': info['usage']
        }
        
        return render_template('index.html', prediction_result=prediction_result)

    except Exception as e:
        return render_template('index.html', error=f'An error occurred. Details: {e}')

if __name__ == '__main__':
    app.run(debug=True)