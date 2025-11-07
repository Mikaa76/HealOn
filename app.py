# Importing essential libraries
from flask import Flask, render_template, request, jsonify
from types import SimpleNamespace
import pickle
import numpy as np
import cv2

# importing all my model inferences ( that use trained model to predict output )
from inferences.tuberculosis import predict_image
from inferences.health_risk import predict_hs
from inferences.med_guess import find_med
from inferences.med_suggest import  recommend_by_symp,recommend_by_medicine
from inferences.perscription import add_record, fetch_all_sorted_by_date
from inferences.ocr import run_ocr



app = Flask(__name__)

# This is basically what page to load when user visits specific url

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/health-risk')
def hlth_form():
	return render_template('form-health.html')

@app.route('/tb')
def tb_form():
	return render_template('form-tb.html')

@app.route('/medulla')
def medulla():
	return render_template('medguess.html')

@app.route('/perscription')
def perscription():
	return render_template('perscription.html')


# Onward are apis that will get and return data to frontend 

# This is one is for tb detection
@app.route('/result', methods=['POST'])

def predict_tb():
    image = request.files['file']
    label, conf = predict_image(image)
    if label == "Tuberculosis":
        label = 1
    else:
        label = 0
    conf = round(conf*100,2)
    return render_template('result-tb.html',tb = label, conf = conf)


# This one get form data and return diabetes and heart risk score
@app.route('/score', methods=['POST'])

def predict_health():
 info = {
    "age": int(request.form['age']),
    "gender": request.form.get('gender'),
    "blood_group": request.form.get('blood'),
    "bmi": float(request.form['bmi']),
    "systolic_bp": int(request.form['bp']),
    "cholesterol": float(request.form['chol']),
    "chest pain" : float(request.form['chest']),
    "stress_level": int(request.form['stress']),
    "sleep_hours": float(request.form['sleep']),
    "exercise_per_week": int(request.form['exercise']),
    "work_hours": float(request.form['work']),
    "junk_food_per_month": int(request.form['junk']),
    "family_history": int(request.form['fam'])
 }

 diab_risk, heart_risk = predict_hs(info)
 
 return render_template('result-health.html',info = info, diab_risk = diab_risk, heart_risk = heart_risk)


# Medicine name guesser!
@app.route('/guess', methods=['POST'])
def med_guess():
     data = request.get_json()   # read JSON body
     query = data.get("query") 
     results = find_med(query)
     return jsonify(results)

# Alternative finder for med name or sympotoms
@app.route('/altmed', methods=['POST'])
def alt_med():
    data = request.get_json()
    type = data.get("type")
    query = data.get("query")
    if(type == 'name'):
        results = recommend_by_medicine(query)
    else:
        results = recommend_by_symp(query)
    
    return jsonify(results)

# Add or retrieve records 
@app.route('/records', methods=['GET','POST'])
def record():
    # read JSON body
  add = request.form.get('add', '0')  

  if add == '1':
     pid = request.form['id']
     name = request.form['name']
     date = request.form['date']
     remarks = request.form['remarks']
     perscript = request.form['perscription']

     add = add_record(pid,name,date,remarks,perscript)
     return render_template('perscription.html',added = add)

  elif request.method == 'POST':
     data = request.get_json()
     pid = data.get("id")
     records = fetch_all_sorted_by_date(pid)
     return jsonify(records)


# When image is upload in perscription form.. This one extract text from it.
@app.route('/ocr', methods=['POST'])       
def ocr():
    if "file" not in request.files:
        return jsonify({"text": "No file uploaded"}), 400

    file = request.files["file"]

    # Convert to NumPy array
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    text = run_ocr(img)
    return jsonify({"text":text})



if __name__ == '__main__':
	app.run(debug=True)

