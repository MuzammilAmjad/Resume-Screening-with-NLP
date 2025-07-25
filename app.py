from flask import Flask, request, render_template
import joblib
import os
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- Initial Setup ---
app = Flask(__name__)

# --- Load Saved Pipeline ---
pipeline = joblib.load("resume_screening_pipeline.pkl")
model = pipeline['model']
vectorizer = pipeline['vectorizer']
label_encoder = pipeline['label_encoder']

# --- Load Resume PDFs ---
def load_resumes():
    resumes = []
    resume_dir = 'data/data'  
    for category in os.listdir(resume_dir):
        category_path = os.path.join(resume_dir, category)
        if os.path.isdir(category_path):
            for file in os.listdir(category_path):
                if file.endswith('.pdf'):
                    try:
                        file_path = os.path.join(category_path, file)
                        doc = fitz.open(file_path)
                        text = ''
                        for page in doc:
                            text += page.get_text()
                        resumes.append({
                            'filename': file,
                            'category': category,
                            'text': text
                        })
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
    return pd.DataFrame(resumes)

resume_df = load_resumes()

# --- Routes ---
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    user_resume = request.form['resume']
    job_desc = request.form['job_desc']

    # No manual cleaning; use raw input
    resume_vec = vectorizer.transform([user_resume])
    pred_cat = model.predict(resume_vec)[0]
    predicted_category = label_encoder.inverse_transform([pred_cat])[0]

    jd_vec = vectorizer.transform([job_desc])
    resume_vecs = vectorizer.transform(resume_df['text'])
    sims = cosine_similarity(jd_vec, resume_vecs).flatten()
    resume_df['similarity'] = sims
    top_matches = resume_df.sort_values(by='similarity', ascending=False).head(5)

    return render_template(
        'result.html', 
        prediction=predicted_category, 
        matches=top_matches[['filename', 'category', 'similarity']].values.tolist()
    )

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
