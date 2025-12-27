from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

MODEL_PATH = r"C:\Users\GATES\Downloads\Project EE\energy_model.joblib"

print(f"🔍 جاري البحث عن الموديل في المسار: {MODEL_PATH}")

def load_model():
    """تحميل نموذج ML باستخدام joblib"""
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print("✅ تم تحميل موديل Colab بنجاح!")
            print(f"📊 نوع الموديل: {type(model)}")
            
            try:
                if hasattr(model, 'feature_names_in_'):
                    print(f"📋 عدد ميزات الموديل: {len(model.feature_names_in_)}")
                    print(f"📋 أول 5 ميزات: {model.feature_names_in_[:5]}")
            except:
                print("ℹ️ لا يمكن قراءة معلومات الميزات من الموديل")
                
            return model
        else:
            print(f"❌ ملف الموديل غير موجود في المسار: {MODEL_PATH}")
            print(f"📂 محتويات المجلد: {os.listdir(os.path.dirname(MODEL_PATH))}")
            return None
    except Exception as e:
        print(f"❌ خطأ في تحميل الموديل: {e}")
        import traceback
        traceback.print_exc()
        return None

print("\n🔄 جاري تحميل الموديل...")
energy_model = load_model()

# تعريف ثابت سعر الكهرباء من الكود الصغير
KWH_PRICE = 0.28  # سعر الكيلو وات

consumption_history = []
dashboard_history = []

def add_features_flask(df):
    """إضافة الميزات بنفس الطريقة المستخدمة في تدريب الموديل"""
    df_feat = df.copy()
    
    if df_feat.index.name != 'date':
        df_feat['date'] = pd.Timestamp.now()
        df_feat.set_index('date', inplace=True)
    
    df_feat['month'] = df_feat.index.month
    df_feat['quarter'] = df_feat.index.quarter
    df_feat['year'] = df_feat.index.year
    
    df_feat['is_month_end'] = (df_feat.index.is_month_end).astype(int)
    
    total_power = df_feat['Total_active_power']
    avg_sub1 = df_feat['Avg_Sub1']
    avg_sub2 = df_feat['Avg_Sub2']
    avg_sub3 = df_feat['Avg_Sub3']
    
    df_feat['total_sub_ratio'] = total_power / (avg_sub1 + avg_sub2 + avg_sub3 + 1e-10)
    df_feat['sub1_to_total'] = avg_sub1 / (total_power + 1e-10)
    df_feat['sub2_to_total'] = avg_sub2 / (total_power + 1e-10)
    df_feat['sub3_to_total'] = avg_sub3 / (total_power + 1e-10)
    
    for lag in [1, 2, 3, 12]:
        df_feat[f'total_lag_{lag}'] = total_power
        df_feat[f'sub1_lag_{lag}'] = avg_sub1
    
    for window in [3, 6, 12]:
        df_feat[f'total_ma_{window}'] = total_power
    
    df_feat['sin_month'] = np.sin(2 * np.pi * df_feat['month'] / 12)
    df_feat['cos_month'] = np.cos(2 * np.pi * df_feat['month'] / 12)
    
    return df_feat

def calculate_prediction_confidence(prediction_source, device_count, input_vs_calculated_diff):
    """حساب ثقة التنبؤ"""
    confidence = 85  # القاعدة
    
    # زيادة الثقة بناءً على مصدر التنبؤ
    if prediction_source == "موديل ML":
        confidence += 10
    
    # زيادة الثقة بناءً على عدد الأجهزة المدخلة (من الكود الصغير)
    confidence += min(device_count * 2, 10)
    
    # تقليل الثقة إذا كان هناك فرق كبير بين المدخل والمحسوب
    if input_vs_calculated_diff > 0.5:
        confidence -= 15
    elif input_vs_calculated_diff > 0.3:
        confidence -= 10
    elif input_vs_calculated_diff > 0.1:
        confidence -= 5
    
    # التأكد من أن الثقة بين 60-95%
    return min(max(confidence, 60), 95)

def calculate_potential_savings(predicted_consumption, estimated_bill):
    """حساب التوفير المحتمل"""
    # يمكن تحقيق توفير بنسبة 15-25% من خلال تحسين السلوك
    savings_percentage = np.random.uniform(0.15, 0.25)
    potential_savings = estimated_bill * savings_percentage
    
    return round(potential_savings, 2)

@app.route('/submit_consumption', methods=['POST'])
def submit_consumption():
    try:
        if energy_model is None:
            print("⚠️ الموديل غير محمل، سيتم استخدام المعادلة التقديرية")
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'لم يتم إرسال بيانات'}), 400

        # 1. استخراج المدخلات الأساسية
        monthly_input = float(data.get("monthly_consumption_input", 0))
        family_members = data.get("family_members", "3")
        
        # 2. استخراج بيانات الأجهزة من المصفوفة
        devices_list = data.get("devices", [])
        
        # 3. حساب الاستهلاك الحالي للأجهزة (من الكود الصغير)
        device_power_kw = {
            "fridge": 0.15,   # ثلاجة 150 وات
            "tv": 0.08,       # تلفزيون 80 وات
            "iron": 0.8,      # مكواة 800 وات
            "washer": 0.6,    # غسالة 600 وات
            "oven": 1.0,      # فرن 1000 وات
            "heater": 1.5,    # سخان 1500 وات
            "ac": 1.2         # مكيف 1200 وات
        }
        
        # حساب استهلاك كل جهاز (من الكود الصغير)
        device_consumption = {}
        total_device_hours = 0
        
        for device in devices_list:
            device_type = device.get("device_type")
            hours_per_day = float(device.get("hours_per_day", 0))
            
            if device_type in device_power_kw:
                # الاستهلاك الشهري = ساعات × قدرة × 30 يوم
                monthly_kwh = hours_per_day * device_power_kw[device_type] * 30
                # استخدام round كما في الكود الصغير
                device_consumption[device_type] = round(monthly_kwh, 2)
                total_device_hours += hours_per_day
        
        # 4. تجهيز البيانات للموديل (إذا كان محملاً)
        avg_sub1 = device_consumption.get("fridge", 0) + device_consumption.get("oven", 0) + device_consumption.get("washer", 0)
        avg_sub2 = device_consumption.get("heater", 0) + device_consumption.get("iron", 0)
        avg_sub3 = device_consumption.get("ac", 0) + device_consumption.get("tv", 0)
        
        total_active_power = monthly_input
        calculated_total = avg_sub1 + avg_sub2 + avg_sub3
        
        print(f"\n📊 بيانات الإدخال المحسوبة:")
        print(f"  استهلاك شهري مدخل: {monthly_input} kWh")
        print(f"  عدد الأجهزة: {len(devices_list)}")
        print(f"  إجمالي ساعات الأجهزة: {total_device_hours} ساعة/يوم")
        print(f"  استهلاك الأجهزة المحسوب: {calculated_total:.2f} kWh")
        
        # 5. استخدام الموديل أو المعادلة التقديرية
        prediction_source = "معادلة تقديرية"
        
        if energy_model is not None:
            try:
                # إعداد البيانات للموديل
                input_data = {
                    "Total_active_power": total_active_power,
                    "Avg_Sub1": avg_sub1,
                    "Avg_Sub2": avg_sub2,
                    "Avg_Sub3": avg_sub3
                }
                
                input_df = pd.DataFrame([input_data])
                input_df_featured = add_features_flask(input_df)
                
                # التنبؤ بالموديل
                prediction = energy_model.predict(input_df_featured)[0]
                prediction = max(prediction, 0)
                prediction_source = "موديل ML"
                
                print(f"  🔍 الموديل استخدم للتنبؤ")
                
            except Exception as e:
                print(f"  ⚠️ خطأ في استخدام الموديل: {e}")
                # استخدام المعادلة التقديرية من الكود الصغير
                prediction = monthly_input * 1.08  # زيادة 8% كتقدير بسيط (من الكود الصغير)
        else:
            # 6. استخدام معادلة تقديرية ذكية
            # الأساس: الاستهلاك الحالي
            base_prediction = monthly_input
            
            # عامل الموسم
            current_month = datetime.now().month
            seasonal_factor = 1.0
            if current_month in [6, 7, 8]:  # الصيف
                seasonal_factor = 1.15
            elif current_month in [12, 1, 2]:  # الشتاء
                seasonal_factor = 1.10
            
            # عامل الأجهزة
            device_factor = 1.0
            if total_device_hours > 0:
                # كل 10 ساعات أجهزة إضافية تزيد الاستهكار 5%
                device_factor = 1.0 + (total_device_hours / 10) * 0.05
            
            # عامل عدد أفراد الأسرة
            family_factor = 1.0
            try:
                family_count = int(family_members)
                if family_count > 2:
                    family_factor = 1.0 + (family_count - 2) * 0.08
            except:
                pass
            
            # حساب التنبؤ النهائي
            prediction = base_prediction * seasonal_factor * device_factor * family_factor
        
        # 7. ضمان أن التنبؤ منطقي
        if prediction < monthly_input * 0.7:
            prediction = monthly_input * 0.9  # لا ينخفض كثيراً
        elif prediction > monthly_input * 2:
            prediction = monthly_input * 1.3  # لا يرتفع كثيراً
        
        print(f"🎯 التنبؤ النهائي: {prediction:.2f} kWh (المصدر: {prediction_source})")
        
        # 8. حساب القيم الإضافية
        # حساب نسبة التغير (من الكود الصغير)
        change_percent = 0
        if monthly_input > 0:
            change_percent = round(((prediction - monthly_input) / monthly_input * 100), 2)
        
        # حساب الفاتورة المقدرة باستخدام الثابت من الكود الصغير
        estimated_bill = round(prediction * KWH_PRICE, 2)
        
        # حساب التوفير المحتمل (من الكود الصغير)
        potential_savings = round(estimated_bill * 0.15, 2)
        
        # حساب ثقة التنبؤ
        input_vs_calculated_diff = 0
        if monthly_input > 0:
            input_vs_calculated_diff = abs(monthly_input - calculated_total) / monthly_input
        
        confidence = calculate_prediction_confidence(
            prediction_source, 
            len(devices_list), 
            input_vs_calculated_diff
        )
        
        # 9. توليد النصائح
        tips = []
        
        # نصائح بناءً على الأجهزة
        high_consumption_devices = []
        for device_type, consumption in device_consumption.items():
            if consumption > monthly_input * 0.2:  # إذا كان الجهاز يستهلك أكثر من 20% من الاستهلاك
                device_names = {
                    "fridge": "الثلاجة",
                    "ac": "المكيف",
                    "heater": "السخان",
                    "oven": "الفرن",
                    "washer": "الغسالة"
                }
                if device_type in device_names:
                    high_consumption_devices.append(device_names[device_type])
        
        if high_consumption_devices:
            tips.append(f"⚠️ الأجهزة التالية تستهلك طاقة عالية: {', '.join(high_consumption_devices)}. حاول تقليل ساعات تشغيلها.")
        
        # نصائح عامة
        if calculated_total > monthly_input * 1.5 and monthly_input > 0:
            tips.append("📊 ملاحظة: استهلاك الأجهزة المحسوب أعلى من فاتورتك الحالية. قد تكون الأجهزة لا تعمل بكامل طاقتها.")
        
        current_month = datetime.now().month
        if current_month in [6, 7, 8]:
            tips.append("☀️ في الصيف: أغلق النوافذ أثناء تشغيل التكييف، واستخدم المراوح عندما تكون الحرارة معتدلة.")
        elif current_month in [12, 1, 2]:
            tips.append("❄️ في الشتاء: استخدم الستائر الثقيلة ليلاً للحفاظ على الحرارة، وأغلق غرف النوم غير المستخدمة.")
        
        # نصائح توفير الطاقة
        if len(devices_list) > 3:
            tips.append("💡 حاول تشغيل الأجهزة الكبيرة (غسالة، فرن) خارج أوقات الذروة (10 مساءً - 6 صباحًا).")
        
        # نصائح بناءً على الفاتورة
        if estimated_bill > 100:
            tips.append("💰 يمكنك توفير حوالي ${:.2f} شهرياً من خلال إجراءات التوفير البسيطة.".format(potential_savings))
        
        if not tips:
            tips.append("✅ استهلاكك معقول. استمر في مراقبة فاتورة الكهرباء وفصل الأجهزة غير المستخدمة.")
        
        # 10. تجهيز الرد النهائي (دمج البيانات من الكودين)
        response_data = {
            "family_members": family_members,
            "calculated_monthly_kwh": round(calculated_total, 2),
            "user_input_monthly_kwh": round(monthly_input, 2),
            "predicted_next_month": round(float(prediction), 2),
            "devices_entered": devices_list,
            "device_consumption": device_consumption,  # تم حسابها بالفعل مع round
            "prediction_source": prediction_source,
            "نسبة_التغير": change_percent,  # تم حسابها بالفعل مع round
            "estimated_bill": estimated_bill,  # تم حسابها بالفعل مع round
            "potential_savings": potential_savings,  # تم حسابها بالفعل مع round
            "prediction_confidence": confidence,
            "نصائح": tips,
            "تاريخ_التحليل": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # إضافة حقول إضافية لضمان التوافق
            "change_percent": change_percent,
            "kwh_price": KWH_PRICE  # إضافة سعر الكهرباء للشفافية
        }
        
        consumption_history.append(response_data)
        
        if len(consumption_history) > 50:
            consumption_history.pop(0)
        
        return jsonify({
            "message": "✅ تم تحليل استهلاك الطاقة بنجاح",
            "result": response_data
        }), 200
        
    except Exception as e:
        print(f"❌ خطأ في معالجة الطلب: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'حدث خطأ: {str(e)}'}), 500

@app.route('/get_alerts', methods=['POST'])
def get_alerts():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        monthly_input = float(data.get("monthly_consumption_input", 0))
        family_members = data.get("family_members", "3")
        devices_list = data.get("devices", [])
        
        alerts = []
        
        # 1. High overall consumption
        if monthly_input > 500:
            alerts.append({
                "title": "Very High Energy Consumption",
                "message": f"Your monthly consumption ({monthly_input} kWh) is above average.",
                "severity": "high",
                "type": "overall_consumption"
            })
        
        # 2. Check individual devices
        device_mapping = {
            "fridge": "Refrigerator",
            "ac": "Air Conditioner",
            "heater": "Water Heater",
            "washer": "Washing Machine",
            "oven": "Oven",
            "iron": "Iron",
            "tv": "TV"
        }
        
        for device in devices_list:
            device_type = device.get("device_type")
            hours = float(device.get("hours_per_day", 0))
            device_name = device.get("device_name", device_mapping.get(device_type, device_type))
            
            # High usage alerts
            if device_type == "heater" and hours > 3:
                alerts.append({
                    "title": "High Water Heater Usage",
                    "message": f"Water heater running {hours} hours daily.",
                    "device": device_name,
                    "severity": "high",
                    "type": "device_usage"
                })
            
            if device_type == "ac" and hours > 8:
                alerts.append({
                    "title": "Extended AC Operation",
                    "message": f"AC running {hours} hours daily.",
                    "device": device_name,
                    "severity": "medium",
                    "type": "device_usage"
                })
            
            if device_type == "fridge" and hours < 24:
                alerts.append({
                    "title": "Refrigerator Not Running Continuously",
                    "message": "Refrigerator should run 24/7 for food safety.",
                    "device": device_name,
                    "severity": "high",
                    "type": "device_usage"
                })
            
            if device_type == "washer" and hours > 2:
                alerts.append({
                    "title": "High Washing Machine Usage",
                    "message": f"Washing machine running {hours} hours daily.",
                    "device": device_name,
                    "severity": "medium",
                    "type": "device_usage"
                })
            
            if device_type == "oven" and hours > 1.5:
                alerts.append({
                    "title": "High Oven Usage",
                    "message": f"Oven running {hours} hours daily.",
                    "device": device_name,
                    "severity": "medium",
                    "type": "device_usage"
                })
        
        # 3. Check total hours
        total_hours = sum(float(d.get("hours_per_day", 0)) for d in devices_list)
        if total_hours > 50:
            alerts.append({
                "title": "High Total Device Runtime",
                "message": f"Devices running {total_hours:.1f} hours daily combined.",
                "severity": "medium",
                "type": "total_runtime"
            })
        
        # 4. If no critical alerts, add a positive message
        if not alerts:
            alerts.append({
                "title": "Energy Usage Normal",
                "message": "No critical alerts detected. Your energy usage patterns appear normal.",
                "severity": "low",
                "type": "normal"
            })
        
        return jsonify({
            "message": "Alerts generated successfully",
            "total_alerts": len(alerts),
            "alerts": alerts,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 200
        
    except Exception as e:
        print(f"Error generating alerts: {e}")
        return jsonify({'error': f'Error generating alerts: {str(e)}'}), 500

@app.route('/get_dashboard_data', methods=['GET'])
def get_dashboard_data():
    try:
        # جلب آخر سجل من التاريخ
        if not consumption_history:
            return jsonify({
                "error": "لا توجد بيانات كافية"
            }), 404
        
        latest_data = consumption_history[-1]
        
        # استخراج البيانات الأساسية
        monthly_consumption = latest_data.get("user_input_monthly_kwh", 0)
        predicted_next_month = latest_data.get("predicted_next_month", 0)
        change_percent = latest_data.get("نسبة_التغير", 0)
        estimated_bill = latest_data.get("estimated_bill", 0)
        potential_savings = latest_data.get("potential_savings", 0)
        confidence = latest_data.get("prediction_confidence", 85)
        
        # إذا لم تكن القيم موجودة، احسبها باستخدام ثوابت الكود الصغير
        if estimated_bill == 0:
            estimated_bill = round(predicted_next_month * KWH_PRICE, 2)
        
        if potential_savings == 0:
            potential_savings = round(estimated_bill * 0.15, 2)
        
        # البيانات النهائية
        dashboard_data = {
            "monthly_consumption": round(monthly_consumption, 2),
            "predicted_next_month": round(predicted_next_month, 2),
            "change_from_last_month": change_percent,
            "estimated_bill": estimated_bill,
            "potential_savings": potential_savings,
            "prediction_confidence": confidence,
            "last_updated": latest_data.get("تاريخ_التحليل", ""),
            "prediction_source": latest_data.get("prediction_source", "معادلة تقديرية"),
            "kwh_price": KWH_PRICE
        }
        
        return jsonify({
            "message": "✅ تم جلب بيانات Dashboard بنجاح",
            "data": dashboard_data
        }), 200
        
    except Exception as e:
        print(f"❌ خطأ في جلب بيانات Dashboard: {e}")
        return jsonify({'error': f'حدث خطأ: {str(e)}'}), 500

@app.route('/update_dashboard', methods=['POST'])
def update_dashboard():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'لم يتم إرسال بيانات'}), 400
        
        # تحديث أو إضافة سجل جديد
        dashboard_record = {
            "monthly_consumption": data.get("monthly_consumption", 0),
            "predicted_next_month": data.get("predicted_next_month", 0),
            "change_percent": data.get("change_percent", 0),
            "prediction_source": data.get("prediction_source", "معادلة تقديرية"),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "kwh_price": KWH_PRICE
        }
        
        dashboard_history.append(dashboard_record)
        
        # الاحتفاظ بأخر 100 سجل فقط
        if len(dashboard_history) > 100:
            dashboard_history.pop(0)
        
        return jsonify({
            "message": "✅ تم تحديث بيانات Dashboard بنجاح",
            "record": dashboard_record,
            "total_records": len(dashboard_history)
        }), 200
        
    except Exception as e:
        print(f"❌ خطأ في تحديث Dashboard: {e}")
        return jsonify({'error': f'حدث خطأ: {str(e)}'}), 500

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({
        "total_records": len(consumption_history),
        "history": consumption_history[-10:],
        "dashboard_history": dashboard_history[-10:],
        "kwh_price": KWH_PRICE
    }), 200

@app.route('/features_info', methods=['GET'])
def get_features_info():
    features_info = {
        "Avg_Sub1": "استهلاك المطبخ والغسيل (ثلاجة، فرن، غسالة)",
        "Avg_Sub2": "استهلاك التدفئة والكوي (سخان، مكواة)",
        "Avg_Sub3": "استهلاك التكييف والترفيه (تكييف، تلفزيون)",
        "Total_active_power": "إجمالي الاستهلاك الشهري المدخل من المستخدم"
    }
    
    model_info = {
        "model_loaded": energy_model is not None,
        "model_type": str(type(energy_model).__name__) if energy_model else "غير محمل",
        "kwh_price": KWH_PRICE
    }
    
    if energy_model and hasattr(energy_model, 'feature_names_in_'):
        model_info["total_features"] = len(energy_model.feature_names_in_)
        model_info["first_5_features"] = energy_model.feature_names_in_[:5].tolist()
    
    return jsonify({
        "model_info": model_info,
        "feature_descriptions": features_info
    }), 200

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global consumption_history, dashboard_history
    count_consumption = len(consumption_history)
    count_dashboard = len(dashboard_history)
    consumption_history = []
    dashboard_history = []
    
    return jsonify({
        "message": f"✅ تم مسح {count_consumption} سجل استهلاك و {count_dashboard} سجل داشبورد"
    }), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "✅ Energy Prediction API is Live",
        "model_status": "✅ محمل وجاهز" if energy_model else "❌ غير محمل",
        "total_consumption_records": len(consumption_history),
        "total_dashboard_records": len(dashboard_history),
        "kwh_price": KWH_PRICE,
        "endpoints": {
            "submit_consumption": "POST /submit_consumption - إرسال بيانات الاستهلاك للتنبؤ",
            "get_dashboard_data": "GET /get_dashboard_data - جلب بيانات الداشبورد",
            "update_dashboard": "POST /update_dashboard - تحديث بيانات الداشبورد",
            "get_alerts": "POST /get_alerts - جلب التنبيهات",
            "history": "GET /history - عرض سجل العمليات",
            "features_info": "GET /features_info - معلومات عن الموديل والميزات",
            "clear_history": "POST /clear_history - مسح السجل"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": energy_model is not None,
        "timestamp": datetime.now().isoformat(),
        "consumption_history_count": len(consumption_history),
        "dashboard_history_count": len(dashboard_history),
        "kwh_price": KWH_PRICE
    }), 200

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 تشغيل Energy Prediction API")
    print(f"📁 المسار الحالي: {os.getcwd()}")
    print(f"🔧 موديل الطاقة: {'✅ محمل' if energy_model else '❌ غير محمل'}")
    print(f"💰 سعر الكيلو وات: ${KWH_PRICE}")
    print(f"📊 سجلات الاستهلاك: {len(consumption_history)}")
    print(f"📈 سجلات الداشبورد: {len(dashboard_history)}")
    print(f"🌐 السيرفر يعمل على: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)