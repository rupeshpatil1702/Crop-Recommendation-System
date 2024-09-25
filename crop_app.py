# crop_app.py
# Importing necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
from flask import Flask, render_template, request

# Load  dataset from CSV
data = pd.read_csv('Crop_recommendation.csv')

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

joblib.dump(model, 'crop_app.pkl')

app = Flask(__name__)

# Route 
@app.route('/')
def home():
    return render_template('Home_1.html')

@app.route('/Predict')
def prediction():
    return render_template('Index.html')

@app.route('/form', methods=["POST"])
def brain():
    model = joblib.load('crop_app.pkl')

    Nitrogen = float(request.form['Nitrogen'])
    Phosphorus = float(request.form['Phosphorus'])
    Potassium = float(request.form['Potassium'])
    Temperature = float(request.form['Temperature'])
    Humidity = float(request.form['Humidity'])
    Ph = float(request.form['ph'])
    Rainfall = float(request.form['Rainfall'])

    values = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]
    if 0 < Ph <= 14 and Temperature < 100 and Humidity > 0:
        arr = [values]
        prediction = model.predict(arr)
        final_prediction = prediction[0]

        return render_template('prediction.html', prediction=final_prediction)
    else:
        return "Sorry... Error in entered values in the form. Please check the values and fill it again."

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
