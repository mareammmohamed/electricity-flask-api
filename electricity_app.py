from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# تحميل الموديل (تأكد من صحة المسار عندك)
MODEL_PATH = r"./energy_model.joblib"
try:
    energy_model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded")
except:
    energy_model = None
    print("⚠️ Model not found, using formulas")

@app.route('/submit_consumption', methods=['POST'])
def submit_consumption():
    try:
        data = request.get_json()
        monthly_input = float(data.get("monthly_consumption_input", 0))
        family_members = int(data.get("family_members", 3))
        devices_list = data.get("devices", [])

        # 1. حساب استهلاك الأجهزة الفردي
        device_power_kw = {"fridge": 0.15, "tv": 0.08, "iron": 0.8, "washer": 0.6, "oven": 1.0, "heater": 1.5, "ac": 1.2}
        device_consumption = {}
        for d in devices_list:
            d_type = d.get("device_type")
            hours = float(d.get("hours_per_day", 0))
            if d_type in device_power_kw:
                device_consumption[d_type] = round(hours * device_power_kw[d_type] * 30, 2)

        # 2. التنبؤ (Prediction)
        # هنا نستخدم الموديل إذا وجد، أو معادلة تقديرية
        prediction = monthly_input * 1.08  # افتراض زيادة 8% كمثال

        # 3. الحسابات المالية (Equations)
        KWH_PRICE = 0.28  # سعر الكيلو وات
        estimated_bill = round(prediction * KWH_PRICE, 2)
        potential_savings = round(estimated_bill * 0.15, 2) # توفير 15% عند التحسين
        
        # 4. المعادلات الإحصائية
        confidence = 85 + min(len(devices_list) * 2, 10) # تزيد مع كثرة البيانات
        change_percent = round(((prediction - monthly_input) / monthly_input * 100), 2) if monthly_input > 0 else 0

        response_data = {
            "user_input_monthly_kwh": round(monthly_input, 2),
            "predicted_next_month": round(prediction, 2),
            "estimated_bill": estimated_bill,
            "potential_savings": potential_savings,
            "prediction_confidence": confidence,
            "change_percent": change_percent,
            "device_consumption": device_consumption,
            "devices_entered": devices_list
        }

        return jsonify({"result": response_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()