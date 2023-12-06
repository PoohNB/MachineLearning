from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load models and scalers
model_path1 = "models/removed_outlier/"
with open(model_path1+'scaler.pkl', 'rb') as file:
    car_scaler1 = pickle.load(file)
with open(model_path1+'car_price.model', 'rb') as file:
    car_model1 = pickle.load(file)


# Model input details
model_input = [
    {"name": "engine (CC)", "type": "int", "default": 1248},
    {"name": "max_power (bhp)", "type": "float", "default": 92.31},
    {"name": "mileage (kmpl)", "type": "float", "default": 19.37},
    {"name": "year", "type": "int", "default": 2015},
]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_values = []
        error_message = None

        for input_info in model_input:
            input_name = input_info["name"]
            input_type = input_info["type"]
            user_input = request.form.get(input_name)

            if user_input == '':
                user_input = input_info["default"]
            try:
                # Convert input to the specified type
                if input_type == "float":
                    user_input = float(user_input)
                elif input_type == "int":
                    user_input = int(user_input)

                input_values.append(user_input) 
      
            except:
                error_message = f"Invalid input for {input_name}. Please enter a valid number."
                break
        
        if error_message:
            return render_template('index.html', model_input=model_input, error_message=error_message)

        # Predict car price using both 
        
        
        input_array = np.array(input_values).reshape(1, -1)

        scaled_input1 = car_scaler1.transform(input_array)
        car_price1 = car_model1.predict(scaled_input1)[0]


        return render_template('result.html', car_price1=round(np.exp(car_price1)), input_values=input_values, model_input=model_input)

    return render_template('index.html', model_input=model_input)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
