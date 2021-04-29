import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask_bootstrap import Bootstrap

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb')) # loading the trained model

@app.route('/') # Homepage
def home():
    return render_template('index.html')
    

@app.route('/connection',methods=['POST'])
def connection():
    if request.method == 'POST':
        text1 = request.form.get("t1")
        exp=[text1]
        result=model.predict(exp)
        print(result)
        return render_template('index.html', prediction_text=result[0])
    else:
        return "Unsuccessful"
     # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)