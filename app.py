import io
import json
import os
import sqlite3
from pathlib import Path
import time

import joblib
import pandas as pd
import requests
from flask import Flask, jsonify, redirect, render_template, request, send_file, send_from_directory, session, flash, url_for
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "mannina-maga-dev-secret-v3")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.environ.get("DATABASE_PATH", DATA_DIR / "mannina_maga.db"))
MODEL_PATH = BASE_DIR / "models" / "yield_model.joblib"
MODEL_META_PATH = BASE_DIR / "models" / "model_metadata.json"

CROP_PROFILES = {
    "Rice":        {"temp": (22, 34), "rain": (1000, 2500), "humidity": (60, 90), "ph": (5.0, 7.0), "n": 80, "p": 40, "k": 40, "base": 4.2},
    "Wheat":       {"temp": (10, 25), "rain": (300, 900),   "humidity": (40, 70), "ph": (6.0, 7.5), "n": 70, "p": 35, "k": 35, "base": 3.4},
    "Maize":       {"temp": (18, 30), "rain": (500, 1200),  "humidity": (40, 75), "ph": (5.5, 7.5), "n": 65, "p": 35, "k": 40, "base": 4.5},
    "Cotton":      {"temp": (21, 35), "rain": (500, 1200),  "humidity": (40, 70), "ph": (5.8, 8.0), "n": 60, "p": 30, "k": 45, "base": 2.2},
    "Soybean":     {"temp": (20, 30), "rain": (500, 1000),  "humidity": (45, 75), "ph": (6.0, 7.5), "n": 40, "p": 40, "k": 40, "base": 2.8},
    "Groundnut":   {"temp": (20, 32), "rain": (400, 1000),  "humidity": (40, 70), "ph": (5.5, 7.0), "n": 35, "p": 35, "k": 45, "base": 2.3},
    "Sugarcane":   {"temp": (20, 35), "rain": (750, 1500),  "humidity": (50, 80), "ph": (6.0, 8.0), "n": 90, "p": 50, "k": 60, "base": 70.0},
    "Jowar":       {"temp": (25, 32), "rain": (400, 800),   "humidity": (35, 65), "ph": (5.5, 8.0), "n": 45, "p": 25, "k": 30, "base": 2.5},
    "Bajra":       {"temp": (25, 35), "rain": (250, 650),   "humidity": (30, 60), "ph": (5.5, 7.5), "n": 40, "p": 22, "k": 28, "base": 2.1},
    "Ragi":        {"temp": (20, 30), "rain": (500, 1000),  "humidity": (45, 75), "ph": (5.0, 7.5), "n": 45, "p": 25, "k": 30, "base": 2.4},
    "Pigeon Pea":  {"temp": (20, 35), "rain": (600, 1000),  "humidity": (40, 70), "ph": (5.0, 7.5), "n": 30, "p": 35, "k": 30, "base": 1.7},
    "Chickpea":    {"temp": (15, 30), "rain": (300, 650),   "humidity": (30, 60), "ph": (6.0, 8.0), "n": 25, "p": 35, "k": 30, "base": 1.8},
    "Sunflower":   {"temp": (20, 30), "rain": (400, 750),   "humidity": (35, 65), "ph": (6.0, 7.5), "n": 45, "p": 30, "k": 40, "base": 1.9},
    "Onion":       {"temp": (13, 30), "rain": (350, 700),   "humidity": (45, 75), "ph": (6.0, 7.5), "n": 70, "p": 40, "k": 60, "base": 18.0},
    "Tomato":      {"temp": (18, 30), "rain": (400, 800),   "humidity": (45, 75), "ph": (5.5, 7.5), "n": 75, "p": 45, "k": 65, "base": 28.0},
}

CROP_LABELS = {
    "en": {
        "Rice":"Rice", "Wheat":"Wheat", "Maize":"Maize", "Cotton":"Cotton", "Soybean":"Soybean",
        "Groundnut":"Groundnut", "Sugarcane":"Sugarcane", "Jowar":"Jowar (Sorghum)", "Bajra":"Bajra (Pearl Millet)",
        "Ragi":"Ragi (Finger Millet)", "Pigeon Pea":"Tur / Red Gram", "Chickpea":"Chickpea / Bengal Gram",
        "Sunflower":"Sunflower", "Onion":"Onion", "Tomato":"Tomato"
    },
    "hi": {
        "Rice":"धान", "Wheat":"गेहूं", "Maize":"मक्का", "Cotton":"कपास", "Soybean":"सोयाबीन",
        "Groundnut":"मूंगफली", "Sugarcane":"गन्ना", "Jowar":"ज्वार", "Bajra":"बाजरा",
        "Ragi":"रागी", "Pigeon Pea":"अरहर / तूर", "Chickpea":"चना",
        "Sunflower":"सूरजमुखी", "Onion":"प्याज", "Tomato":"टमाटर"
    },
    "kn": {
        "Rice":"ಭತ್ತ", "Wheat":"ಗೋಧಿ", "Maize":"ಮೆಕ್ಕೆಜೋಳ", "Cotton":"ಹತ್ತಿ", "Soybean":"ಸೋಯಾಬೀನ್",
        "Groundnut":"ಕಡಲೆಕಾಯಿ", "Sugarcane":"ಕಬ್ಬು", "Jowar":"ಜೋಳ", "Bajra":"ಸಜ್ಜೆ",
        "Ragi":"ರಾಗಿ", "Pigeon Pea":"ತೊಗರಿ", "Chickpea":"ಕಡಲೆ",
        "Sunflower":"ಸೂರ್ಯಕಾಂತಿ", "Onion":"ಈರುಳ್ಳಿ", "Tomato":"ಟೊಮ್ಯಾಟೊ"
    }
}

TRANSLATIONS = {
    "en": {
        "dashboard":"Home", "weather":"Weather", "yield":"Crop Yield", "advisor":"Crop Advisor", "soil":"Soil Health", "analytics":"Trends", "about":"About", "logout":"Logout",
        "welcome":"Welcome", "farm_overview":"YOUR FARM TODAY", "home_intro":"See the most useful information for your farm in one place.", "quick_actions":"What would you like to do?",
        "farm_area":"Farm Area", "main_crop":"Main Crop", "soil_health":"Soil Health", "weather_status":"Weather", "check_weather":"Check Weather", "weather_desc":"See live weather and a simple 4-day forecast.",
        "predict_yield":"Estimate Yield", "yield_desc":"Use weather and soil values to estimate crop yield.", "crop_advice":"Choose Crop", "crop_desc":"Find crops that suit your farm conditions.", "soil_check":"Check Soil", "soil_desc":"Understand N, P, K and pH in simple words.",
        "live_weather":"Live Weather", "location":"Village / City", "search":"Show Weather", "temperature":"Temperature", "humidity":"Humidity", "rain_today":"Rain today", "wind":"Wind", "forecast":"Next days", "weather_source":"Weather data: Open-Meteo", "weather_unavailable":"Weather is unavailable right now. Please try again.",
        "yield_title":"Crop Yield Estimate", "yield_intro":"Enter your farm values. Use Live Weather to fill weather values automatically.", "use_live_weather":"Use Live Weather", "crop":"Crop", "area":"Farm Area (acres)", "rainfall":"Season rainfall (mm)", "soil_ph":"Soil pH", "nitrogen":"Nitrogen (N)", "phosphorus":"Phosphorus (P)", "potassium":"Potassium (K)", "predict":"Estimate Yield", "yield_per_ha":"Yield per hectare", "total_production":"Estimated production", "farm_match":"Farm match", "download_pdf":"Download PDF Report", "why_result":"Why this result?", "weather_fit":"Weather fit", "ph_fit":"Soil pH fit", "npk_fit":"NPK balance", "ml_active":"scikit-learn model active", "estimate_note":"This is decision support, not a guarantee. Local field conditions can change actual yield.",
        "advisor_title":"Crop Advisor", "advisor_intro":"Enter farm conditions to see the three best matching crops.", "recommend":"Recommend Crops", "top_matches":"Best Matches",
        "soil_title":"Soil Health", "soil_intro":"Enter your soil test values. If you do not know them, use a local soil test / Soil Health Card before applying fertilizer.", "analyze_soil":"Check Soil", "npk_health":"NPK Health", "recommendations":"Simple Recommendations", "download_soil":"Download Soil PDF",
        "trends_title":"Farm Trends", "trends_intro":"Simple charts to understand how crop yield can change over time.",
        "about_title":"About Mannina Maga", "about_intro":"A farmer-friendly crop intelligence project using simple web technology.", "data_sources":"Data Sources", "technology":"Technology", "data_note":"Crop production source selected: Government of India OGD. Historical rainfall source selected: IMD. Current weather: Open-Meteo.", "model_note":"The included scikit-learn model is a starter model so the app works immediately. A script is included to retrain it with a cleaned official merged dataset.",
        "language":"Language", "simple_mode":"Simple Mode", "online":"Online", "offline":"Offline", "report":"Farm PDF", "report_desc":"Download a simple farm summary you can save or share.", "download_farm_report":"Download Farm Report",
        "signin":"Sign In", "register":"Create Farm Profile", "email":"Email", "password":"Password", "name":"Farmer Name", "already_registered":"Already registered?", "new_here":"New here?", "create_account":"Create Account", "signin_intro":"Sign in to open your farm dashboard.", "register_intro":"Add your basic farm details. You can change data later.",
        "acre":"acres", "good":"Good", "not_available":"Not available"
    },
    "hi": {
        "dashboard":"होम", "weather":"मौसम", "yield":"फसल उपज", "advisor":"फसल सलाह", "soil":"मिट्टी स्वास्थ्य", "analytics":"रुझान", "about":"जानकारी", "logout":"लॉग आउट",
        "welcome":"नमस्ते", "farm_overview":"आज आपका खेत", "home_intro":"अपने खेत की जरूरी जानकारी एक ही जगह देखें।", "quick_actions":"आप क्या करना चाहते हैं?",
        "farm_area":"खेत का क्षेत्र", "main_crop":"मुख्य फसल", "soil_health":"मिट्टी स्वास्थ्य", "weather_status":"मौसम", "check_weather":"मौसम देखें", "weather_desc":"लाइव मौसम और 4 दिन का सरल पूर्वानुमान देखें।",
        "predict_yield":"उपज अनुमान", "yield_desc":"मौसम और मिट्टी के आधार पर फसल की उपज का अनुमान लगाएं।", "crop_advice":"फसल चुनें", "crop_desc":"अपने खेत के लिए उपयुक्त फसलें देखें।", "soil_check":"मिट्टी जांचें", "soil_desc":"N, P, K और pH को आसान भाषा में समझें।",
        "live_weather":"लाइव मौसम", "location":"गांव / शहर", "search":"मौसम दिखाएं", "temperature":"तापमान", "humidity":"नमी", "rain_today":"आज की बारिश", "wind":"हवा", "forecast":"अगले दिन", "weather_source":"मौसम डेटा: Open-Meteo", "weather_unavailable":"अभी मौसम जानकारी नहीं मिल रही है। फिर प्रयास करें।",
        "yield_title":"फसल उपज अनुमान", "yield_intro":"अपने खेत की जानकारी भरें। मौसम की जानकारी अपने आप भरने के लिए लाइव मौसम का उपयोग करें।", "use_live_weather":"लाइव मौसम भरें", "crop":"फसल", "area":"खेत का क्षेत्र (एकड़)", "rainfall":"मौसम की वर्षा (मिमी)", "soil_ph":"मिट्टी pH", "nitrogen":"नाइट्रोजन (N)", "phosphorus":"फॉस्फोरस (P)", "potassium":"पोटैशियम (K)", "predict":"उपज बताएं", "yield_per_ha":"प्रति हेक्टेयर उपज", "total_production":"अनुमानित उत्पादन", "farm_match":"खेत मिलान", "download_pdf":"PDF रिपोर्ट डाउनलोड करें", "why_result":"यह परिणाम क्यों?", "weather_fit":"मौसम मिलान", "ph_fit":"मिट्टी pH मिलान", "npk_fit":"NPK संतुलन", "ml_active":"scikit-learn मॉडल सक्रिय", "estimate_note":"यह केवल निर्णय सहायता है, गारंटी नहीं। वास्तविक खेत की स्थिति से उपज बदल सकती है।",
        "advisor_title":"फसल सलाह", "advisor_intro":"अपने खेत की स्थिति भरें और तीन सबसे उपयुक्त फसलें देखें।", "recommend":"फसल सुझाएं", "top_matches":"सबसे अच्छे विकल्प",
        "soil_title":"मिट्टी स्वास्थ्य", "soil_intro":"मिट्टी जांच के मान भरें। यदि मान नहीं पता हैं, खाद डालने से पहले स्थानीय मिट्टी जांच / Soil Health Card करवाएं।", "analyze_soil":"मिट्टी जांचें", "npk_health":"NPK स्वास्थ्य", "recommendations":"सरल सलाह", "download_soil":"मिट्टी PDF डाउनलोड करें",
        "trends_title":"खेत रुझान", "trends_intro":"समय के साथ फसल उपज में बदलाव समझने के लिए सरल चार्ट।",
        "about_title":"Mannina Maga के बारे में", "about_intro":"सरल वेब तकनीक पर बना किसान-अनुकूल फसल जानकारी प्रोजेक्ट।", "data_sources":"डेटा स्रोत", "technology":"तकनीक", "data_note":"फसल उत्पादन के लिए चयनित स्रोत: भारत सरकार OGD। ऐतिहासिक वर्षा: IMD। वर्तमान मौसम: Open-Meteo।", "model_note":"ऐप तुरंत चलाने के लिए एक starter scikit-learn मॉडल शामिल है। आधिकारिक साफ किए गए डेटा से दोबारा ट्रेन करने की स्क्रिप्ट भी दी गई है।",
        "language":"भाषा", "simple_mode":"सरल मोड", "online":"ऑनलाइन", "offline":"ऑफलाइन", "report":"खेत PDF", "report_desc":"एक सरल खेत रिपोर्ट डाउनलोड करें जिसे आप सेव या शेयर कर सकते हैं।", "download_farm_report":"खेत रिपोर्ट डाउनलोड करें",
        "signin":"साइन इन", "register":"खेत प्रोफाइल बनाएं", "email":"ईमेल", "password":"पासवर्ड", "name":"किसान का नाम", "already_registered":"पहले से खाता है?", "new_here":"नए हैं?", "create_account":"खाता बनाएं", "signin_intro":"अपने खेत का डैशबोर्ड खोलने के लिए साइन इन करें।", "register_intro":"अपने खेत की मूल जानकारी भरें।", "acre":"एकड़", "good":"अच्छा", "not_available":"उपलब्ध नहीं"
    },
    "kn": {
        "dashboard":"ಮುಖಪುಟ", "weather":"ಹವಾಮಾನ", "yield":"ಬೆಳೆ ಇಳುವರಿ", "advisor":"ಬೆಳೆ ಸಲಹೆ", "soil":"ಮಣ್ಣಿನ ಆರೋಗ್ಯ", "analytics":"ಪ್ರವೃತ್ತಿ", "about":"ಮಾಹಿತಿ", "logout":"ಲಾಗ್ ಔಟ್",
        "welcome":"ನಮಸ್ಕಾರ", "farm_overview":"ಇಂದಿನ ನಿಮ್ಮ ಹೊಲ", "home_intro":"ನಿಮ್ಮ ಹೊಲದ ಮುಖ್ಯ ಮಾಹಿತಿಯನ್ನು ಒಂದೇ ಜಾಗದಲ್ಲಿ ನೋಡಿ.", "quick_actions":"ನೀವು ಏನು ಮಾಡಲು ಬಯಸುತ್ತೀರಿ?",
        "farm_area":"ಹೊಲದ ವಿಸ್ತೀರ್ಣ", "main_crop":"ಮುಖ್ಯ ಬೆಳೆ", "soil_health":"ಮಣ್ಣಿನ ಆರೋಗ್ಯ", "weather_status":"ಹವಾಮಾನ", "check_weather":"ಹವಾಮಾನ ನೋಡಿ", "weather_desc":"ಲೈವ್ ಹವಾಮಾನ ಮತ್ತು ಸರಳ 4 ದಿನಗಳ ಮುನ್ಸೂಚನೆ ನೋಡಿ.",
        "predict_yield":"ಇಳುವರಿ ಅಂದಾಜು", "yield_desc":"ಹವಾಮಾನ ಮತ್ತು ಮಣ್ಣಿನ ಮಾಹಿತಿಯಿಂದ ಬೆಳೆ ಇಳುವರಿ ಅಂದಾಜಿಸಿ.", "crop_advice":"ಬೆಳೆ ಆಯ್ಕೆ", "crop_desc":"ನಿಮ್ಮ ಹೊಲಕ್ಕೆ ಹೊಂದುವ ಬೆಳೆಗಳನ್ನು ನೋಡಿ.", "soil_check":"ಮಣ್ಣು ಪರಿಶೀಲಿಸಿ", "soil_desc":"N, P, K ಮತ್ತು pH ಅನ್ನು ಸರಳವಾಗಿ ಅರ್ಥಮಾಡಿಕೊಳ್ಳಿ.",
        "live_weather":"ಲೈವ್ ಹವಾಮಾನ", "location":"ಗ್ರಾಮ / ನಗರ", "search":"ಹವಾಮಾನ ತೋರಿಸಿ", "temperature":"ತಾಪಮಾನ", "humidity":"ಆರ್ದ್ರತೆ", "rain_today":"ಇಂದಿನ ಮಳೆ", "wind":"ಗಾಳಿ", "forecast":"ಮುಂದಿನ ದಿನಗಳು", "weather_source":"ಹವಾಮಾನ ಡೇಟಾ: Open-Meteo", "weather_unavailable":"ಈಗ ಹವಾಮಾನ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "yield_title":"ಬೆಳೆ ಇಳುವರಿ ಅಂದಾಜು", "yield_intro":"ನಿಮ್ಮ ಹೊಲದ ಮಾಹಿತಿಯನ್ನು ನಮೂದಿಸಿ. ಹವಾಮಾನ ಮೌಲ್ಯಗಳನ್ನು ಸ್ವಯಂ ತುಂಬಲು ಲೈವ್ ಹವಾಮಾನ ಬಳಸಿ.", "use_live_weather":"ಲೈವ್ ಹವಾಮಾನ ತುಂಬಿ", "crop":"ಬೆಳೆ", "area":"ಹೊಲದ ವಿಸ್ತೀರ್ಣ (ಎಕರೆ)", "rainfall":"ಋತು ಮಳೆ (ಮಿಮೀ)", "soil_ph":"ಮಣ್ಣಿನ pH", "nitrogen":"ನೈಟ್ರೋಜನ್ (N)", "phosphorus":"ಫಾಸ್ಫರಸ್ (P)", "potassium":"ಪೊಟ್ಯಾಸಿಯಂ (K)", "predict":"ಇಳುವರಿ ಅಂದಾಜಿಸಿ", "yield_per_ha":"ಪ್ರತಿ ಹೆಕ್ಟೇರ್ ಇಳುವರಿ", "total_production":"ಅಂದಾಜು ಉತ್ಪಾದನೆ", "farm_match":"ಹೊಲ ಹೊಂದಾಣಿಕೆ", "download_pdf":"PDF ವರದಿ ಡೌನ್‌ಲೋಡ್", "why_result":"ಈ ಫಲಿತಾಂಶ ಏಕೆ?", "weather_fit":"ಹವಾಮಾನ ಹೊಂದಾಣಿಕೆ", "ph_fit":"ಮಣ್ಣಿನ pH ಹೊಂದಾಣಿಕೆ", "npk_fit":"NPK ಸಮತೋಲನ", "ml_active":"scikit-learn ಮಾದರಿ ಸಕ್ರಿಯ", "estimate_note":"ಇದು ನಿರ್ಧಾರ ಸಹಾಯ ಮಾತ್ರ, ಖಾತರಿ ಅಲ್ಲ. ನೈಜ ಹೊಲದ ಪರಿಸ್ಥಿತಿಯಿಂದ ಇಳುವರಿ ಬದಲಾಗಬಹುದು.",
        "advisor_title":"ಬೆಳೆ ಸಲಹೆ", "advisor_intro":"ಹೊಲದ ಪರಿಸ್ಥಿತಿಯನ್ನು ನಮೂದಿಸಿ ಮತ್ತು ಅತ್ಯುತ್ತಮ ಮೂರು ಬೆಳೆಗಳನ್ನು ನೋಡಿ.", "recommend":"ಬೆಳೆ ಸೂಚಿಸಿ", "top_matches":"ಉತ್ತಮ ಹೊಂದಾಣಿಕೆಗಳು",
        "soil_title":"ಮಣ್ಣಿನ ಆರೋಗ್ಯ", "soil_intro":"ಮಣ್ಣಿನ ಪರೀಕ್ಷಾ ಮೌಲ್ಯಗಳನ್ನು ನಮೂದಿಸಿ. ತಿಳಿದಿಲ್ಲದಿದ್ದರೆ ರಸಗೊಬ್ಬರ ಬಳಸುವ ಮೊದಲು ಸ್ಥಳೀಯ ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ / Soil Health Card ಮಾಡಿಸಿ.", "analyze_soil":"ಮಣ್ಣು ಪರಿಶೀಲಿಸಿ", "npk_health":"NPK ಆರೋಗ್ಯ", "recommendations":"ಸರಳ ಸಲಹೆಗಳು", "download_soil":"ಮಣ್ಣಿನ PDF ಡೌನ್‌ಲೋಡ್",
        "trends_title":"ಹೊಲದ ಪ್ರವೃತ್ತಿಗಳು", "trends_intro":"ಕಾಲಕ್ರಮೇಣ ಬೆಳೆ ಇಳುವರಿ ಹೇಗೆ ಬದಲಾಗಬಹುದು ಎಂಬುದನ್ನು ಸರಳ ಚಾರ್ಟ್‌ಗಳಲ್ಲಿ ನೋಡಿ.",
        "about_title":"ಮಣ್ಣಿನ ಮಗ ಬಗ್ಗೆ", "about_intro":"ನಮ್ಮ ಮಣ್ಣು ಮತ್ತು ರೈತರಿಗಾಗಿ ಸರಳ ವೆಬ್ ತಂತ್ರಜ್ಞಾನದಿಂದ ನಿರ್ಮಿಸಿದ ರೈತ ಸ್ನೇಹಿ ಬೆಳೆ ಮಾಹಿತಿ ಯೋಜನೆ.", "data_sources":"ಡೇಟಾ ಮೂಲಗಳು", "technology":"ತಂತ್ರಜ್ಞಾನ", "data_note":"ಬೆಳೆ ಉತ್ಪಾದನೆಗೆ ಆಯ್ದ ಮೂಲ: ಭಾರತ ಸರ್ಕಾರ OGD. ಐತಿಹಾಸಿಕ ಮಳೆ: IMD. ಪ್ರಸ್ತುತ ಹವಾಮಾನ: Open-Meteo.", "model_note":"ಅಪ್ಲಿಕೇಶನ್ ತಕ್ಷಣ ಕೆಲಸ ಮಾಡಲು starter scikit-learn ಮಾದರಿ ಸೇರಿಸಲಾಗಿದೆ. ಅಧಿಕೃತ ಸ್ವಚ್ಛ ಡೇಟಾದಿಂದ ಮರುತರಬೇತಿ ಮಾಡಲು ಸ್ಕ್ರಿಪ್ಟ್ ಕೂಡ ಇದೆ.",
        "language":"ಭಾಷೆ", "simple_mode":"ಸರಳ ಮೋಡ್", "online":"ಆನ್‌ಲೈನ್", "offline":"ಆಫ್‌ಲೈನ್", "report":"ಹೊಲ PDF", "report_desc":"ಉಳಿಸಿಕೊಳ್ಳಲು ಅಥವಾ ಹಂಚಿಕೊಳ್ಳಲು ಸರಳ ಹೊಲ ವರದಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ.", "download_farm_report":"ಹೊಲ ವರದಿ ಡೌನ್‌ಲೋಡ್",
        "signin":"ಸೈನ್ ಇನ್", "register":"ಹೊಲ ಪ್ರೊಫೈಲ್ ರಚಿಸಿ", "email":"ಇಮೇಲ್", "password":"ಪಾಸ್‌ವರ್ಡ್", "name":"ರೈತನ ಹೆಸರು", "already_registered":"ಈಗಾಗಲೇ ಖಾತೆ ಇದೆಯೇ?", "new_here":"ಹೊಸದಾ?", "create_account":"ಖಾತೆ ರಚಿಸಿ", "signin_intro":"ನಿಮ್ಮ ಹೊಲ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್ ತೆರೆಯಲು ಸೈನ್ ಇನ್ ಮಾಡಿ.", "register_intro":"ನಿಮ್ಮ ಹೊಲದ ಮೂಲ ಮಾಹಿತಿಯನ್ನು ಸೇರಿಸಿ.", "acre":"ಎಕರೆ", "good":"ಚೆನ್ನಾಗಿದೆ", "not_available":"ಲಭ್ಯವಿಲ್ಲ"
    }
}

_MODEL = None

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                location TEXT DEFAULT '',
                crop TEXT DEFAULT 'Rice',
                area REAL DEFAULT 5
            )
        """)

def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    with db() as conn:
        return conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()

def require_login():
    user = current_user()
    if user is None:
        # A stale browser cookie can contain an old user_id even when the
        # current database does not contain that user anymore.
        session.pop("user_id", None)
        return redirect(url_for("login"))
    return None

def tr(key):
    lang = session.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))

def crop_label(crop):
    lang = session.get("lang", "en")
    return CROP_LABELS.get(lang, CROP_LABELS["en"]).get(crop, crop)

@app.context_processor
def inject_globals():
    return {
        "user": current_user(),
        "tr": tr,
        "crop_label": crop_label,
        "current_lang": session.get("lang", "en"),
    }

def score_range(value, low, high):
    if low <= value <= high:
        return 1.0
    distance = min(abs(value - low), abs(value - high))
    scale = max(high - low, 1)
    return max(0.0, 1.0 - distance / scale)

def crop_score(profile, rainfall, temp, humidity, ph, n, p, k):
    parts = [
        score_range(rainfall, *profile["rain"]),
        score_range(temp, *profile["temp"]),
        score_range(humidity, *profile["humidity"]),
        score_range(ph, *profile["ph"]),
        max(0, 1 - abs(n-profile["n"]) / max(profile["n"], 1)),
        max(0, 1 - abs(p-profile["p"]) / max(profile["p"], 1)),
        max(0, 1 - abs(k-profile["k"]) / max(profile["k"], 1)),
    ]
    return round(sum(parts) / len(parts) * 100, 1)

def factor_breakdown(profile, rainfall, temp, humidity, ph, n, p, k):
    weather = (score_range(rainfall,*profile["rain"]) + score_range(temp,*profile["temp"]) + score_range(humidity,*profile["humidity"])) / 3
    soil_ph = score_range(ph,*profile["ph"])
    nutrient = (max(0,1-abs(n-profile["n"])/max(profile["n"],1)) + max(0,1-abs(p-profile["p"])/max(profile["p"],1)) + max(0,1-abs(k-profile["k"])/max(profile["k"],1))) / 3
    return [
        {"label": tr("weather_fit"), "score": round(weather*100)},
        {"label": tr("ph_fit"), "score": round(soil_ph*100)},
        {"label": tr("npk_fit"), "score": round(nutrient*100)},
    ]

def load_model():
    global _MODEL
    if _MODEL is None and MODEL_PATH.exists():
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL

def model_metadata():
    try:
        return json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"model":"Unavailable", "training_data":"Unknown"}

def predict_ml(crop, rainfall, temp, humidity, n, p, k, ph):
    model = load_model()
    if model is None:
        return None
    frame = pd.DataFrame([{
        "crop":crop, "rainfall_mm":rainfall, "temp_c":temp, "humidity_pct":humidity,
        "nitrogen":n, "phosphorus":p, "potassium":k, "ph":ph,
    }])
    return max(0.05, float(model.predict(frame)[0]))

WEATHER_CODES = {
    0:"Clear sky", 1:"Mostly clear", 2:"Partly cloudy", 3:"Cloudy",
    45:"Fog", 48:"Fog", 51:"Light drizzle", 53:"Drizzle", 55:"Heavy drizzle",
    61:"Light rain", 63:"Rain", 65:"Heavy rain", 80:"Rain showers", 81:"Rain showers", 82:"Heavy showers",
    95:"Thunderstorm", 96:"Thunderstorm", 99:"Thunderstorm",
}

def weather_label(code):
    try: return WEATHER_CODES.get(int(code), "Weather")
    except Exception: return "Weather"

def get_live_weather(location):
    location = (location or "").strip()
    if not location:
        return None
    geo = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name":location,"count":1,"language":"en","format":"json","countryCode":"IN"}, timeout=6)
    geo.raise_for_status()
    results = geo.json().get("results") or []
    if not results:
        return None
    place = results[0]
    lat, lon = place["latitude"], place["longitude"]
    forecast = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude":lat, "longitude":lon,
        "current":"temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m",
        "daily":"weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
        "timezone":"auto", "forecast_days":4,
    }, timeout=7)
    forecast.raise_for_status()
    data = forecast.json()
    cur = data.get("current") or {}
    daily = data.get("daily") or {}
    days=[]
    for i, date in enumerate(daily.get("time", [])):
        days.append({
            "date":date,
            "label":weather_label((daily.get("weather_code") or [None]*4)[i]),
            "max":(daily.get("temperature_2m_max") or [None]*4)[i],
            "min":(daily.get("temperature_2m_min") or [None]*4)[i],
            "rain":(daily.get("precipitation_sum") or [None]*4)[i],
            "rain_chance":(daily.get("precipitation_probability_max") or [None]*4)[i],
        })
    name = ", ".join([x for x in [place.get("name"), place.get("admin1")] if x])
    return {
        "location":name or location,
        "latitude":lat, "longitude":lon,
        "temperature":cur.get("temperature_2m"),
        "humidity":cur.get("relative_humidity_2m"),
        "wind":cur.get("wind_speed_10m"),
        "condition":weather_label(cur.get("weather_code")),
        "rainfall_today": (daily.get("precipitation_sum") or [cur.get("rain",0)])[0],
        "days":days,
        "source":"Open-Meteo",
    }

def safe_weather(location):
    for attempt in range(3):
        try:
            return get_live_weather(location)
        except Exception as e:
            print(
                f"WEATHER ERROR attempt {attempt + 1} for {location}: {repr(e)}",
                flush=True
            )
            time.sleep(1.5)

    return None

def pdf_response(title, rows, filename, note=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Mannina MagaTitle", parent=styles["Title"], textColor=colors.HexColor("#155e3b"), alignment=TA_CENTER, spaceAfter=12)
    story=[Paragraph(title, title_style), Paragraph("Mannina Maga - Farmer Report", styles["Heading2"]), Spacer(1,6)]
    table_data=[["Item","Value"]] + [[str(a),str(b)] for a,b in rows]
    table=Table(table_data, colWidths=[62*mm, 92*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dcfce7")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#14532d")),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#d1d5db")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("PADDING",(0,0),(-1,-1),7),
    ]))
    story += [table, Spacer(1,10)]
    if note:
        story.append(Paragraph(note, styles["BodyText"]))
    story.append(Spacer(1,8))
    story.append(Paragraph("Data notes: current weather uses Open-Meteo. Historical production/rainfall sources selected for official retraining are Government of India OGD and IMD. Farm recommendations are decision support only.", styles["BodyText"]))
    story.append(Spacer(1,10))
    story.append(Paragraph("Developed by Subodh Desai", ParagraphStyle("DeveloperCredit", parent=styles["BodyText"], alignment=TA_CENTER, textColor=colors.HexColor("#64748b"), fontSize=9)))
    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name=filename)

@app.route("/")
def home():
    if current_user() is not None:
        return redirect(url_for("dashboard"))
    session.pop("user_id", None)
    return redirect(url_for("login"))

@app.route("/language", methods=["POST"])
def set_language():
    lang=request.form.get("lang","en")
    if lang in TRANSLATIONS: session["lang"]=lang
    nxt=request.form.get("next") or url_for("dashboard")
    return redirect(nxt if nxt.startswith("/") else url_for("dashboard"))

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method=="POST":
        name=request.form.get("name","").strip(); email=request.form.get("email","").strip().lower(); password=request.form.get("password","")
        location=request.form.get("location","").strip(); crop=request.form.get("crop","Rice"); area=request.form.get("area","5")
        if not name or not email or len(password)<4:
            flash("Please enter name, email and a password of at least 4 characters.","error")
            return render_template("register.html",crops=CROP_PROFILES.keys())
        try:
            with db() as conn:
                conn.execute("INSERT INTO users(name,email,password,location,crop,area) VALUES(?,?,?,?,?,?)",(name,email,generate_password_hash(password),location,crop,float(area or 5)))
            flash("Account created. Please sign in.","success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError: flash("That email is already registered.","error")
        except ValueError: flash("Farm area must be a number.","error")
    return render_template("register.html",crops=CROP_PROFILES.keys())

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        email=request.form.get("email","").strip().lower(); password=request.form.get("password","")
        with db() as conn: u=conn.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone()
        if u and check_password_hash(u["password"],password):
            session["user_id"]=u["id"]
            return redirect(url_for("dashboard"))
        flash("Invalid email or password.","error")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    guard=require_login()
    if guard: return guard
    u=current_user(); weather=safe_weather(u["location"])
    metrics={"farm_area":u["area"],"crop":u["crop"],"soil_health":84}
    return render_template("dashboard.html",metrics=metrics,weather=weather)

@app.route("/weather")
def weather():
    guard=require_login()
    if guard: return guard
    location=request.args.get("location", current_user()["location"] or "")
    weather_data=safe_weather(location) if location else None
    attempted=bool(request.args.get("location"))
    return render_template("weather.html",weather=weather_data,location=location,attempted=attempted)

@app.route("/api/weather")
def api_weather():
    u = current_user()
    if u is None:
        session.pop("user_id", None)
        return jsonify({"error":"login required"}),401
    location=request.args.get("location") or u["location"]
    try:
        data=get_live_weather(location)
        if not data: return jsonify({"error":"location not found"}),404
        return jsonify(data)
    except Exception:
        return jsonify({"error":"weather service unavailable"}),503

@app.route("/yield-predictor", methods=["GET","POST"])
def yield_predictor():
    guard=require_login()
    if guard: return guard
    u=current_user(); result=None
    form={"crop":u["crop"],"area":u["area"],"rainfall":850,"temp":27,"humidity":62,"n":70,"p":40,"k":45,"ph":6.6}
    if request.method=="POST":
        try:
            for key in ["area","rainfall","temp","humidity","n","p","k","ph"]: form[key]=float(request.form.get(key,form[key]))
            form["crop"]=request.form.get("crop",form["crop"])
            profile=CROP_PROFILES[form["crop"]]
            suitability=crop_score(profile,form["rainfall"],form["temp"],form["humidity"],form["ph"],form["n"],form["p"],form["k"])
            ml_yield=predict_ml(form["crop"],form["rainfall"],form["temp"],form["humidity"],form["n"],form["p"],form["k"],form["ph"])
            if ml_yield is None:
                ml_yield=profile["base"]*(0.55+suitability/100*0.55)
                model_name="Fallback farm score"
            else:
                model_name="Gradient Boosting - scikit-learn"
            yield_per_ha=round(float(ml_yield),2)
            hectares=form["area"]*0.404686
            production=round(yield_per_ha*hectares,2)
            result={"yield_per_ha":yield_per_ha,"production":production,"suitability":suitability,"model":model_name,"explain":factor_breakdown(profile,form["rainfall"],form["temp"],form["humidity"],form["ph"],form["n"],form["p"],form["k"])}
            session["last_yield"]={"form":form,"result":result}
        except (ValueError,KeyError): flash("Please enter valid values.","error")
    return render_template("yield.html",crops=CROP_PROFILES.keys(),form=form,result=result,model_meta=model_metadata())

@app.route("/reports/yield.pdf")
def yield_pdf():
    guard=require_login()
    if guard: return guard
    saved=session.get("last_yield")
    if not saved:
        flash("Please calculate yield first.","error"); return redirect(url_for("yield_predictor"))
    f=saved["form"]; r=saved["result"]; u=current_user()
    rows=[("Farmer",u["name"]),("Location",u["location"] or "Not provided"),("Crop",crop_label(f["crop"])),("Farm area",f"{f['area']} acres"),("Rainfall",f"{f['rainfall']} mm"),("Temperature",f"{f['temp']} C"),("Humidity",f"{f['humidity']}%"),("Soil pH",f["ph"]),("N-P-K",f"{f['n']} - {f['p']} - {f['k']}"),("Estimated yield",f"{r['yield_per_ha']} t/ha"),("Estimated production",f"{r['production']} tonnes"),("Farm match",f"{r['suitability']}%"),("Model",r["model"])]
    return pdf_response("Crop Yield Report",rows,"mannina-maga-yield-report.pdf","Use this report as decision support. Actual yield depends on local field conditions, crop variety and farm practices.")

@app.route("/crop-advisor", methods=["GET","POST"])
def crop_advisor():
    guard=require_login()
    if guard: return guard
    results=[]; values={"rainfall":800,"temp":26,"humidity":60,"n":60,"p":40,"k":45,"ph":6.7}
    if request.method=="POST":
        try:
            values={k:float(request.form.get(k,v)) for k,v in values.items()}; ranked=[]
            for crop,profile in CROP_PROFILES.items(): ranked.append((crop,crop_score(profile,values["rainfall"],values["temp"],values["humidity"],values["ph"],values["n"],values["p"],values["k"])))
            results=sorted(ranked,key=lambda x:x[1],reverse=True)[:3]
        except ValueError: flash("Please enter valid numbers.","error")
    return render_template("crop.html",values=values,results=results)

@app.route("/soil-lab", methods=["GET","POST"])
def soil_lab():
    guard=require_login()
    if guard: return guard
    values={"n":70,"p":45,"k":50,"ph":6.5}; report=None
    if request.method=="POST":
        try:
            values={k:float(request.form.get(k,v)) for k,v in values.items()}; n_score=min(values["n"]/80*100,100); p_score=min(values["p"]/50*100,100); k_score=min(values["k"]/60*100,100); ph=values["ph"]
            if ph<5.5: ph_label="Acidic"; ph_note="Soil is acidic. Confirm treatment with a local soil expert before applying lime."
            elif ph>7.8: ph_label="Alkaline"; ph_note="Soil is alkaline. Add organic matter and confirm treatment locally."
            else: ph_label="Good"; ph_note="pH is suitable for many crops."
            notes=[]
            if values["n"]<45: notes.append("Nitrogen looks low. Confirm with a soil test before adding nitrogen fertilizer.")
            if values["p"]<25: notes.append("Phosphorus looks low. Confirm with a soil test before adding phosphorus fertilizer.")
            if values["k"]<30: notes.append("Potassium looks low. Confirm with a soil test before adding potash.")
            if not notes: notes.append("NPK values look reasonably balanced for this simple check.")
            report={"n_score":round(n_score),"p_score":round(p_score),"k_score":round(k_score),"ph_label":ph_label,"ph_note":ph_note,"notes":notes}
            session["last_soil"]={"values":values,"report":report}
        except ValueError: flash("Please enter valid soil values.","error")
    return render_template("soil.html",values=values,report=report)

@app.route("/reports/soil.pdf")
def soil_pdf():
    guard=require_login()
    if guard: return guard
    saved=session.get("last_soil")
    if not saved:
        flash("Please check soil first.","error"); return redirect(url_for("soil_lab"))
    v=saved["values"]; r=saved["report"]; u=current_user()
    rows=[("Farmer",u["name"]),("Location",u["location"] or "Not provided"),("Nitrogen",v["n"]),("Phosphorus",v["p"]),("Potassium",v["k"]),("Soil pH",v["ph"]),("N score",f"{r['n_score']}%"),("P score",f"{r['p_score']}%"),("K score",f"{r['k_score']}%"),("pH status",r["ph_label"]),("Recommendation","; ".join(r["notes"]))]
    return pdf_response("Soil Health Report",rows,"mannina-maga-soil-report.pdf","Fertilizer quantity should be confirmed using a local soil test and agronomy recommendation.")

@app.route("/reports/farm.pdf")
def farm_pdf():
    guard=require_login()
    if guard: return guard
    u=current_user(); w=safe_weather(u["location"])
    rows=[("Farmer",u["name"]),("Location",u["location"] or "Not provided"),("Main crop",crop_label(u["crop"])),("Farm area",f"{u['area']} acres")]
    if w:
        rows += [("Current weather",w["condition"]),("Temperature",f"{w['temperature']} C"),("Humidity",f"{w['humidity']}%"),("Rain today",f"{w['rainfall_today']} mm"),("Weather source",w["source"])]
    meta=model_metadata(); rows += [("Yield model",meta.get("model","scikit-learn")),("Model training",meta.get("training_data","starter dataset"))]
    return pdf_response("Farm Summary",rows,"mannina-maga-farm-report.pdf")

@app.route("/analytics")
def analytics():
    guard=require_login()
    if guard: return guard
    months=["Jan","Feb","Mar","Apr","May","Jun"]
    series={"Wheat":[2.3,2.5,2.7,2.9,3.05,3.15],"Rice":[2.6,2.8,3.0,3.2,3.35,3.45],"Maize":[2.15,2.35,2.55,2.75,2.9,3.05]}
    return render_template("analytics.html",months=months,series=series)

@app.route("/about")
def about():
    guard=require_login()
    if guard: return guard
    return render_template("about.html",model_meta=model_metadata())

@app.route("/offline")
def offline(): return render_template("offline.html")

@app.route("/manifest.webmanifest")
def manifest(): return send_from_directory(app.static_folder,"manifest.webmanifest",mimetype="application/manifest+json")

@app.route("/service-worker.js")
def service_worker():
    response=send_from_directory(app.static_folder,"sw.js",mimetype="application/javascript"); response.headers["Service-Worker-Allowed"]="/"; return response


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(app.static_folder, "icons/icon-192.png", mimetype="image/png")

@app.route("/health")
def health(): return jsonify({"status":"ok","app":"Mannina Maga","model":MODEL_PATH.exists()}),200

init_db()

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000)); debug=os.environ.get("FLASK_DEBUG","0")=="1"
    app.run(host="0.0.0.0",port=port,debug=debug)
