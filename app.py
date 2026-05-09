import os, json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib

DATA_PATH = 'laptop_data2.csv'

app     = Flask(__name__)
model   = joblib.load('model.pkl')
encoders = joblib.load('encoders.pkl')

with open('meta.json') as f:
    meta = json.load(f)

df_raw = pd.read_csv(DATA_PATH)

@app.route('/')
def index():
    return render_template('index.html', meta=meta)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        d = request.json
        row = {
            'brand':            encoders['brand'].transform([d['brand']])[0],
            'ram_gb':           int(d['ram_gb']),
            'storage_gb':       float(d['storage_gb']),
            'storage_type':     encoders['storage_type'].transform([d['storage_type']])[0],
            'screen_size':      float(d['screen_size']),
            'os':               encoders['os'].transform([d['os']])[0],
            'processor_brand':  encoders['processor_brand'].transform([d['processor_brand']])[0],
            'processor_series': encoders['processor_series'].transform([d['processor_series']])[0],
            'generation':       int(d['generation']),
        }
        X = pd.DataFrame([row])[meta['features']]
        price = float(model.predict(X)[0])

        # Find similar laptops
        df_enc = df_raw.copy()
        similar = df_raw[
            (df_raw['ram_gb'] == int(d['ram_gb'])) &
            (df_raw['processor_series'].str.lower() == d['processor_series'].lower())
        ][['brand','price','ram_gb','storage_gb','storage_type','processor_series','generation']]\
         .sort_values('price').head(5).to_dict(orient='records')

        return jsonify({
            'predicted_price': round(price, -2),
            'price_formatted': f"₹{round(price,-2):,.0f}",
            'confidence_low':  f"₹{round(price * 0.90, -2):,.0f}",
            'confidence_high': f"₹{round(price * 1.10, -2):,.0f}",
            'similar': similar
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analytics')
def analytics():
    # Avg price by brand
    brand_avg = df_raw.groupby('brand')['price'].mean().round(0).sort_values(ascending=False)

    # Price by RAM
    ram_avg = df_raw.groupby('ram_gb')['price'].mean().round(0).sort_values()

    # Price by processor series
    proc_avg = df_raw.groupby('processor_series')['price'].median().round(0).sort_values(ascending=False)

    # Storage type distribution
    storage_dist = df_raw['storage_type'].value_counts()

    # OS distribution
    os_dist = df_raw['os'].value_counts()

    # Price distribution buckets
    buckets = ['<30k', '30-50k', '50-70k', '70-1L', '1L+']
    counts = [
        len(df_raw[df_raw['price'] < 30000]),
        len(df_raw[(df_raw['price'] >= 30000) & (df_raw['price'] < 50000)]),
        len(df_raw[(df_raw['price'] >= 50000) & (df_raw['price'] < 70000)]),
        len(df_raw[(df_raw['price'] >= 70000) & (df_raw['price'] < 100000)]),
        len(df_raw[df_raw['price'] >= 100000]),
    ]

    return jsonify({
        'brand_avg':    {'labels': brand_avg.index.tolist(), 'values': brand_avg.values.tolist()},
        'ram_avg':      {'labels': [str(r) for r in ram_avg.index.tolist()], 'values': ram_avg.values.tolist()},
        'proc_avg':     {'labels': proc_avg.index.tolist(), 'values': proc_avg.values.tolist()},
        'storage_dist': {'labels': storage_dist.index.tolist(), 'values': storage_dist.values.tolist()},
        'os_dist':      {'labels': os_dist.index.tolist(), 'values': os_dist.values.tolist()},
        'price_dist':   {'labels': buckets, 'values': counts},
        'model_results': meta['model_results'],
        'feature_importance': meta['feature_importance'],
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
