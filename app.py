from warnings import filterwarnings
filterwarnings("ignore")
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from markupsafe import Markup

import sqlite3
import os
import random
from datetime import datetime
import json

import pickle
# CROP RECOMENDATION MODEL
cr = pickle.load(open('models/RandomForest.pkl', 'rb'))

# CROP YIELD MODEL
forest = pickle.load(open('models/yield_rf.pkl', 'rb'))  # yield

# FERTILIZER RECOMENDATION MODEL
model = pickle.load(open('models/classifier.pkl','rb'))
ferti = pickle.load(open('models/xgb.pkl','rb'))

# DISEASE DETECTION
from disease import disease_dic
from fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from model import ResNet9

from googletrans import Translator

# Create translator instance
translator = Translator()

chat_history = []
chat_history_kn = []
chat_history_hi = []
import google.generativeai as genai
genai.configure(api_key='api_key')
gemini_model = genai.GenerativeModel('gemini-2.0-flash')
chat = gemini_model.start_chat(history=[])

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

#######################################################################################################

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Initialize database
def init_db():
    conn = sqlite3.connect('krishimitra.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Community questions table
    c.execute('''CREATE TABLE IF NOT EXISTS questions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  title TEXT NOT NULL,
                  content TEXT NOT NULL,
                  user_id INTEGER,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Answers table
    c.execute('''CREATE TABLE IF NOT EXISTS answers
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  content TEXT NOT NULL,
                  question_id INTEGER,
                  user_id INTEGER,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (question_id) REFERENCES questions (id),
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

init_db()

import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import csv
import io

# Add allowed file extensions for CSV upload
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/priceprediction', methods=['GET', 'POST'])
def priceprediction():
    if 'user_id' not in session:
        return redirect(url_for('auth'))
    
    if request.method == 'POST':
        # Check if file was uploaded
        if 'csv_file' not in request.files:
            flash('No file selected!', 'error')
            return redirect(request.url)
        
        file = request.files['csv_file']
        
        if file.filename == '':
            flash('No file selected!', 'error')
            return redirect(request.url)
        print(f"\n\n\n\n {file} \n\n\n\n")
        print(file.filename)                 # shows 'apple.csv'
        pcrop = file.filename.rsplit('.', 1)[0]   # gives 'apple'
        print(pcrop)
        from check import check_stat
        highest_price_value,highest_price_day=check_stat(file.filename)
        print(f"highest_price_day :\n\n{highest_price_day}\n\n")
        print(f"highest_price_value :\n\n{highest_price_value:.2f}\n\n")
        high_month=highest_price_day
        high_price=int(highest_price_value)
        session['pcrop'] = pcrop
        session['priceday'] = high_month
        session['pricerate'] = high_price

        return render_template('priceprediction.html',high_price=high_price,high_month=high_month,filename=file.filename) 

                    
        
    
    return render_template('priceprediction.html')

# Database helper functions
def get_db_connection():
    conn = sqlite3.connect('krishimitra.db')
    conn.row_factory = sqlite3.Row
    return conn

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':
        user_input = request.form['query']
        if user_input == "hi":
            chat_history.append([user_input, "Hi , How are you this is Mental Health related chatbot assistance"])
        # Get response from Gemini AI model
        else:
            gemini_response = chat.send_message(user_input+"for this  query give response  and i know providing medical advice online is risky so dont show excess contents (note: provide in html format content)")
            data = gemini_response.text
            data = data.replace("```html", "")
            data = data.replace("```", "")
            chat_history.append([user_input, data])
            
        return render_template('chatbot.html', chat_history=chat_history)
    return render_template('chatbot.html')
    
@app.route('/analyseh', methods=['GET', 'POST'])
def analyseh():
    if request.method == 'POST':
        user_input = request.form['query']
        if user_input.lower() in ["hi", "hello", "नमस्ते"]:
            response_text = "नमस्ते, आज मैं आपकी कैसे सहायता कर सकता हूँ?"

            
        else:
            from_lang = 'hi'
            to_lang = 'en'
            text_to_translate = translator.translate(user_input, src=from_lang, dest=to_lang)
            text = text_to_translate.text  
            
            gemini_response = chat.send_message(text+"for this  query give response  and i know providing medical advice online is risky so dont show excess contents (note: provide in html format content)")
            data = gemini_response.text
            data = data.replace("```html", "")
            response = data.replace("```", "")
            
            print(f"Bot: {response}")

            from_lang = 'en'
            to_lang = 'hi'
            text_to_translate = translator.translate(response, src=from_lang, dest=to_lang)
            text = text_to_translate.text

            # TTS(text, 'hi')
            # chat_history_hi.append([user_input, data])
            chat_history_hi.append([user_input, text])
            
            
        return render_template('chatboth.html', chat_history_hi=chat_history_hi)
    return render_template('chatboth.html')

@app.route('/analysek', methods=['GET', 'POST'])
def analysek():
    if request.method == 'POST':
        user_input = request.form['query']
        if user_input.lower() in ["hi", "hello", "ನಮಸ್ತೆ", "ನಮಸ್ಕಾರ"]:
            response_text = "ಹಾಯ್, ನಾನು ಇಂದು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?"
        else:
            from_lang = 'kn'
            to_lang = 'en'
            text_to_translate = translator.translate(user_input, src=from_lang, dest=to_lang)
            text = text_to_translate.text  


            gemini_response = chat.send_message(text+"for this  query give response  and i know providing medical advice online is risky so dont show excess contents (note: provide in html format content)")
            data = gemini_response.text
            data = data.replace("```html", "")
            response     = data.replace("```", "")
            

            # response = chatbot_response(text)
            print(f"Bot: {response}")

            from_lang = 'en'
            to_lang = 'kn'
            text_to_translate = translator.translate(response, src=from_lang, dest=to_lang)
            text = text_to_translate.text

            

            chat_history_kn.append([user_input, text])
            return render_template('chatbotk.html',chat_history_kn=chat_history_kn)
        
            
            
        return render_template('chatbotk.html', chat_history_kn=chat_history_kn)
    return render_template('chatbotk.html')


@app.route('/auth')
def auth():
    return render_template('auth.html')

@app.route('/calender')
def calender():
    return render_template('calender.html')

@app.route('/weatherforecast', methods=['GET', 'POST'])
def weatherforecast():
    if 'user_id' not in session:
        return redirect(url_for('auth'))

    from weather_forecast import get_forecast, all_city_names
    city_query = (request.values.get('city') or '').strip()
    result = get_forecast(city_query) if city_query else None
    return render_template('weatherforecast.html',
                           result=result,
                           city_query=city_query,
                           city_names=all_city_names())

    

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                    (username, email, password))
        conn.commit()
        flash('Registration successful! Please login.', 'success')
    except sqlite3.IntegrityError:
        flash('Username or email already exists!', 'error')
    finally:
        conn.close()
    
    return redirect(url_for('auth'))

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', 
                       (username, password)).fetchone()
    conn.close()
    
    if user:
        session['user_id'] = user['id']
        session['username'] = user['username']
        flash('Login successful!', 'success')
        return redirect(url_for('index'))
    else:
        flash('Invalid credentials!', 'error')
        return redirect(url_for('auth'))

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))



@ app.route('/yieldprediction', methods=['GET','POST'])
def yieldprediction():
    title = 'yeild predicted'

    if request.method == 'POST':
        state = request.form['stt']
        district = request.form['city']
        year = 2024 #request.form['year']
        season = request.form['season']
        crop = request.form['crop']
        Temperature = request.form['Temperature']
        humidity = request.form['humidity']
        soilmoisture = request.form['soilmoisture']
        area = request.form['area']

        out_1 = forest.predict([[float(state),
                                 float(district),
                                 float(year),
                                 float(season),
                                 float(crop),
                                 float(Temperature),
                                 float(humidity),
                                 float(soilmoisture),
                                 float(area)]])
        if out_1[0] > 1000:
            ans=f" {out_1[0]:.2f} Kg"
        elif out_1[0] > 100:
            ans=f" {out_1[0]:.2f} Quintal"
        else:
            ans=f" {out_1[0]:.2f} Tons"

        print(f"\n\n\n\n {ans} \n\n\n\n")

        session['yield'] = ans
        


        return render_template('yieldprediction.html', prediction=ans, title=title)

    return render_template('yieldprediction.html', title=title)

@app.route('/croprecommend', methods=['GET', 'POST'])
def croprecommend():
    if 'user_id' not in session:
        return redirect(url_for('auth'))
    
    if request.method == 'POST':
        # Get form data with correct parameter names
        N = request.form['nitrogen']
        P = request.form['phosphorous']
        K = request.form['pottasium']
        ph = request.form['ph']
        rainfall = request.form['rainfall']
        hum = request.form['hum']
        temp = request.form['temp']

        # Convert to numpy array for prediction
        data = np.array([[N, P, K, temp, hum, ph, rainfall]])
        
        # Get prediction
        my_prediction = cr.predict(data)
        final_prediction = my_prediction[0]

        # Predict probabilities for all crops
        probs = cr.predict_proba(data)[0]  # Get probability distribution

        # Get top 3 crop indices
        top_3_indices = np.argsort(probs)[-3:][::-1]  # Sort in descending order

        # Get crop names based on indices
        crop_names = cr.classes_  # Assuming the classifier stores crop names in .classes_
        top_3_crops = [crop_names[i] for i in top_3_indices]

        # Store in session and render template
        session['crop'] = top_3_crops
        return render_template('croprecommend.html', prediction=top_3_crops)
    
    # GET request - show the form
    return render_template('croprecommend.html')

@app.route('/fertilizerrecommend', methods=['GET', 'POST'])
def fertilizerrecommend():
    if 'user_id' not in session:
        return redirect(url_for('auth'))
    
    if request.method == 'POST':
        # Get form data with correct parameter names from your working code
        temp = request.form.get('temp')
        humi = request.form.get('humid')
        mois = request.form.get('mois')
        soil = request.form.get('soil')
        crop = request.form.get('crop')
        nitro = request.form.get('nitro')
        pota = request.form.get('pota')
        phosp = request.form.get('phos')
        
        # Convert to float and create input array
        input_data = [float(temp), float(humi), float(mois), float(soil), 
                     float(crop), float(nitro), float(pota), float(phosp)]
        
        # Get prediction using your existing models
        indexx = model.predict([input_data])[0]
        res = ferti.classes_[indexx]
        
        # Store in session
        session['fer'] = res
        
        # Render template with results
        return render_template('fertilizerrecommend.html', 
                             res=res, 
                             indexx=str(indexx),
                             temp=temp, hum=humi, moi=mois, n=nitro, k=pota, p=phosp)
    
    # GET request - show the form
    return render_template('fertilizerrecommend.html')

@app.route('/pestdetection', methods=['GET', 'POST'])
def pestdetection():
    if 'user_id' not in session:
        return redirect(url_for('auth'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files.get('file')
        if not file:
            return render_template('pestdetection.html')
        
        # try:
        # Read the image file
        img = file.read()
        
        # Get prediction using your existing function
        prediction = predict_image(img)
        
        # Convert to readable format using your disease dictionary
        prediction_text = Markup(str(disease_dic[prediction]))
        
        # Save the uploaded file
        filename = f"disease_{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.seek(0)  # Reset file pointer
        file.save(filepath)
        
        # Store in session
        session['disease'] = prediction_text
        
        # Render template with results
        return render_template('pestdetection.html', 
                                prediction=prediction_text,
                                image_url=f'/static/uploads/{filename}',
                                current_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # except Exception as e:
        #     print(f"Error processing image: {e}")
        #     return render_template('pestdetection.html', 
        #                          prediction="Error processing image. Please try again.")
    
    # GET request - show the form
    return render_template('pestdetection.html')

@app.route('/chatbot')
def chatbot():
    if 'user_id' not in session:
        return redirect(url_for('auth'))
    return render_template('chatbot.html')

@app.route('/quiz')
def quiz():
    if 'user_id' not in session:
        return redirect(url_for('auth'))
    return render_template('quiz.html')

@app.route('/community')
def community():
    conn = get_db_connection()
    questions = conn.execute('''
        SELECT q.*, u.username 
        FROM questions q 
        JOIN users u ON q.user_id = u.id 
        ORDER BY q.created_at DESC
    ''').fetchall()
    conn.close()
    return render_template('community.html', questions=questions)

@app.route('/ask_question', methods=['POST'])
def ask_question():
    if 'user_id' not in session:
        return redirect(url_for('auth'))
    
    title = request.form['title']
    content = request.form['content']
    
    conn = get_db_connection()
    conn.execute('INSERT INTO questions (title, content, user_id) VALUES (?, ?, ?)',
                (title, content, session['user_id']))
    conn.commit()
    conn.close()
    
    flash('Question posted successfully!', 'success')
    return redirect(url_for('community'))

@app.route('/answer_question', methods=['POST'])
def answer_question():
    if 'user_id' not in session:
        return redirect(url_for('auth'))
    
    question_id = request.form['question_id']
    content = request.form['content']
    
    conn = get_db_connection()
    conn.execute('INSERT INTO answers (content, question_id, user_id) VALUES (?, ?, ?)',
                (content, question_id, session['user_id']))
    conn.commit()
    conn.close()
    
    flash('Answer posted successfully!', 'success')
    return redirect(url_for('community'))

@app.route('/get_answers/<int:question_id>')
def get_answers(question_id):
    conn = get_db_connection()
    answers = conn.execute('''
        SELECT a.*, u.username 
        FROM answers a 
        JOIN users u ON a.user_id = u.id 
        WHERE a.question_id = ? 
        ORDER BY a.created_at ASC
    ''', (question_id,)).fetchall()
    conn.close()
    
    answers_list = []
    for answer in answers:
        answers_list.append({
            'id': answer['id'],
            'content': answer['content'],
            'username': answer['username'],
            'created_at': answer['created_at']
        })
    
    return jsonify(answers_list)

if __name__ == '__main__':
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)