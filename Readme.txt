LLM-Powered Audio Call Analyzer

This project is a comprehensive system for analyzing customer support calls using cutting-edge machine learning models.

Features:

* Automatic Speech Recognition (ASR) with diarization
* Speaker Role Classification (Agent vs Customer)
* Sentiment Analysis using CardiffNLP's RoBERTa model
* Tonal Emotion Analysis using HuBERT
* LLM-Powered Evaluation using Groq and HuggingFace models
* Contextual Scoring from a predefined rulebook via FAISS
* Interactive Gradio Interface

Supports:

* WhisperX and Assembly AI for transcription + diarization
* Speaker classification using LLaMA 3 via Groq
* Sentiment and tonal analysis for each speaker
* Generates visual charts of analysis
* Retrieves relevant rulebook context using FAISS vector search
* Evaluates overall call quality using an LLM

Contributors:

* Group 15
* M. Annus Shabbir — 24280015
* Eeman Adnan — 24280022
* Talha Nasir — 24280040
* M. Arslan Rafique — 24280064
* Kashaf Gohar — 24280009

Run Locally:

1. Create Virtual Environment

2. Install Dependencies:
   pip install -r requirements.txt

3. Run the Application:
   python llm_project.py

4. Access the Interface:
   Open your browser and navigate to:
   [http://localhost:7860](http://localhost:7860)

Notes:

* Unable to deploy on Hugging Face due to memory issues. It was taking more than 16gb.
* faiss_vectorstore.py is used to generate the FAISS vector store.
