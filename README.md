# Laptop Price Prediction System

This is a Flask-based laptop price prediction web app trained on Flipkart laptop listings.

## Files

- `app.py` — Flask application serving the UI and API.
- `train.py` — training script that creates `model.pkl`, `encoders.pkl`, and `meta.json`.
- `laptop_data2.csv` — dataset used for training and analytics.
- `templates/index.html` — frontend UI template.
- `requirements.txt` — Python dependencies for Vercel.
- `vercel.json` — Vercel deployment configuration.

## Deploy to Vercel

1. Install Vercel CLI if needed:
   ```bash
   npm install -g vercel
   ```

2. Log in to Vercel:
   ```bash
   vercel login
   ```

3. Deploy the app:
   ```bash
   vercel --prod --confirm
   ```

## Run locally

```bash
python app.py
```

Then open `http://localhost:8000`.

## Notes

- Make sure `model.pkl`, `encoders.pkl`, `meta.json`, and `laptop_data2.csv` are present in the project root.
- Vercel will use `@vercel/python` and route all requests to `app.py`.
