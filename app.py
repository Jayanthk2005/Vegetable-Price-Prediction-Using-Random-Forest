from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient
import pandas as pd
import numpy as np
import joblib
import datetime

app = Flask(__name__)

# MongoDB configuration
client = MongoClient('mongodb://localhost:27017/')
db = client['vegetable_prices_db']
feedback_collection = db['feedback']

# Load the trained ML model
model = joblib.load('rf_model.pkl')

# List of supported vegetables with display names
vegetables = [
    {"name": "potato", "display_name": "Potato"},
    {"name": "tomato", "display_name": "Tomato"},
    {"name": "peas", "display_name": "Peas"},
    {"name": "pumpkin", "display_name": "Pumpkin"},
    {"name": "cucumber", "display_name": "Cucumber"},
    {"name": "onion", "display_name": "Onion"},
    {"name": "carrot", "display_name": "Carrot"},
    {"name": "brinjal", "display_name": "Brinjal"},
    {"name": "capsicum", "display_name": "Capsicum"},
    {"name": "cabbage", "display_name": "Cabbage"}
]

# Predict prices for a specific date
def get_prices(date):
    current_month = date.month
    season = (
        "Winter" if current_month in [12, 1, 2] else
        "Summer" if current_month in [3, 4, 5] else
        "Monsoon"
    )

    predictions = []

    for veg in vegetables:
        input_data = pd.DataFrame([{
            'vegetable_' + veg["name"]: 1,
            'season_' + season: 1,
            'month_' + str(current_month): 1,
            'disaster_None': 1,
            'condition_Good': 1
        }])

        # Fill in any missing columns that the model expects
        for col in model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[model.feature_names_in_]

        predicted_price = model.predict(input_data)[0]
        actual_price = predicted_price + np.random.uniform(-5, 5)

        predictions.append({
            "name": veg["name"],
            "display_name": veg["display_name"],
            "predicted_price": round(predicted_price, 2),
            "actual_price": round(actual_price, 2)
        })

    return predictions

# Seasonal recommendations based on the current month
def get_seasonal_recommendations():
    current_month = datetime.datetime.now().month
    season = (
        "Winter" if current_month in [12, 1, 2] else
        "Summer" if current_month in [3, 4, 5] else
        "Monsoon"
    )
    seasonal_recommendations = {
        "Winter": ["Carrot", "Cabbage", "Peas"],
        "Summer": ["Cucumber", "Tomato", "Pumpkin"],
        "Monsoon": ["Brinjal", "Capsicum", "Onion"]
    }
    return seasonal_recommendations.get(season, [])

# Nutritional information for each vegetable
def get_nutrition_info():
    return {
        "potato": {"calories": 77, "fiber": 2.2, "vitamin_c": 19.7},
        "tomato": {"calories": 18, "fiber": 1.2, "vitamin_c": 13.7},
        "peas": {"calories": 81, "fiber": 5.1, "vitamin_c": 40},
        "pumpkin": {"calories": 26, "fiber": 0.5, "vitamin_c": 9},
        "cucumber": {"calories": 16, "fiber": 0.5, "vitamin_c": 2.8},
        "onion": {"calories": 40, "fiber": 1.7, "vitamin_c": 8.1},
        "carrot": {"calories": 41, "fiber": 2.8, "vitamin_c": 7.6},
        "brinjal": {"calories": 25, "fiber": 3, "vitamin_c": 2.2},
        "capsicum": {"calories": 20, "fiber": 1.7, "vitamin_c": 128.3},
        "cabbage": {"calories": 25, "fiber": 2.5, "vitamin_c": 36.6}
    }

# Routes
@app.route('/')
def home():
    vegetables = get_prices(datetime.datetime.now().date())
    seasonal_recommendations = get_seasonal_recommendations()
    return render_template('index.html', vegetables=vegetables, seasonal_recommendations=seasonal_recommendations)

@app.route('/predict', methods=['GET'])
def predict():
    date = request.args.get('date')
    if not date:
        return jsonify({"error": "Date is required"}), 400

    try:
        parsed_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        prices = get_prices(parsed_date)
        return jsonify(prices)
    except ValueError:
        return jsonify({"error": "Invalid date format"}), 400

@app.route('/nutrient')
def nutrient():
    nutrition_info = get_nutrition_info()
    return render_template('nutrient.html', nutrient_info=nutrition_info)

@app.route('/seasonal')
def seasonal():
    seasonal_recommendations = get_seasonal_recommendations()
    return render_template('seasonal.html', seasonal_recommendations=seasonal_recommendations)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        feedback = {
            'name': name,
            'email': email,
            'message': message,
            'timestamp': datetime.datetime.now()
        }
        feedback_collection.insert_one(feedback)
        return render_template('contact.html', success=True)
    
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
