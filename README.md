#  Resume Screening System

This project is an AI-based Resume Screening System that classifies resumes into predefined categories and matches them with job descriptions using Natural Language Processing (NLP) techniques.

---

## Features

-  Trained on labeled resume data (e.g., HR, IT, Data Science)
-  Converts resume text into vectors using TF-IDF
-  Classifies resumes using machine learning (e.g., Logistic Regression)
-  Compares job descriptions with resumes using cosine similarity
-  Ranks resumes based on job description relevance
-  Flask-based web app for deployment
-  Model and vectorizer saved via joblib for reuse

---


##  How it Works

1.  **Training Phase**
   - Load labeled resume data
   - Preprocess text (cleaning, lowercasing, removing stopwords)
   - Convert text into numeric vectors using `TF-IDF`
   - Train a classification model (e.g., Logistic Regression)

2.  **Prediction Phase**
   - Accept new resume input
   - Vectorize and classify the resume into a job category
   - Compare the resume with a job description using cosine similarity
   - Output category + matching score

3.  **Deployment Phase**
   - Flask backend takes user input
   - Sends to model and vectorizer
   - Returns prediction and top matches in HTML interface

---
