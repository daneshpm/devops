import pickle
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return "Welcome to the ML App!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON data from request
        data = request.json
        
        # Ensure 'features' is provided in the request
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400
        
        # Make prediction
        prediction = model.predict([data["features"]])
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

