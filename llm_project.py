import os
import re
import json
import torch
import gradio as gr
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
from pydub import AudioSegment
from collections import defaultdict, Counter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Wav2Vec2FeatureExtractor,
    HubertForSequenceClassification,
    pipeline
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import assemblyai as aai
from groq import Groq
import whisperx

port = int(os.environ.get("PORT", 10000))

uvicorn.run(app, host="0.0.0.0", port=port)

from dotenv import load_dotenv
load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# Sentiment analysis setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment"
).to(device)
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=sentiment_model,
    tokenizer=sentiment_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Tonal analysis setup
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")
hubert_tonal_model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
hubert_tonal_model.eval()

# Rulebook and evaluation setup
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "faiss_langchain_store/faiss_langchain_store",
    embedding_model,
    allow_dangerous_deserialization=True
)

# LLM for evaluation
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    tokenizer="mistralai/Mistral-7B-Instruct-v0.1",
    use_auth_token=HUGGINGFACE_API_TOKEN,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
    return_full_text=False,
    max_new_tokens=512
)

# === UTILITY FUNCTIONS ===
def convert_to_wav(audio_path):
    """Convert any audio file to WAV format."""
    output_path = os.path.splitext(audio_path)[0] + ".wav"
    audio = AudioSegment.from_file(audio_path)
    audio.export(output_path, format="wav")
    return output_path

def transcribe_with_diarization(audio_path):
    """Transcribe audio with speaker diarization using AssemblyAI."""
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        speaker_labels=True
    )
    
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_path)
    
    if transcript.status == "error":
        raise RuntimeError(f"Transcription failed: {transcript.error}")
    
    return transcript

def classify_speakers(transcript_text):
    """Classify speakers as Agent or Customer using Groq LLaMA 3."""
    prompt = """
    You are analyzing a customer support phone call transcript.
    Your job is to label each speaker turn with 'Agent' or 'Customer' only.
    Keep all timestamps and content the same. Replace the speaker label (like A:, B:, C:) with Agent: or Customer:.

    Here is the transcript:
    {transcript}

    Output the updated transcript with correct Agent/Customer roles.
    """.format(transcript=transcript_text)
    
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are an expert in labeling speaker roles in dialogues."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def transcribe_with_diarization_whisperx(audio_path, min_speakers=2, max_speakers=2, model_name="base"):
    """
    Transcribe audio with speaker diarization using WhisperX.
    
    Args:
        audio_path (str): Path to audio file
        min_speakers (int): Minimum number of speakers expected
        max_speakers (int): Maximum number of speakers expected
        model_name (str): Whisper model size (tiny, base, small, medium, large)
    
    Returns:
        dict: Transcription result with speaker assignments
    """
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    batch_size = 4 if device == "cuda" else 2
    
    try:
        # Load model and transcribe
        model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=batch_size)
        
        # Free memory
        del model
        gc.collect()
        
        # Align words
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=device
        )
        result = whisperx.align(
            result["segments"], 
            align_model, 
            metadata, 
            audio, 
            device, 
            return_char_alignments=False
        )
        
        # Free memory
        del align_model
        gc.collect()
        
        # Run diarization
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=os.getenv("HF_AUTH_TOKEN"),
            device=device
        )
        diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        
        # Assign speakers
        final_result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Format segments similar to AssemblyAI output
        segments = []
        for seg in final_result["segments"]:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg.get("speaker", "Unknown"),
                "text": seg["text"].strip()
            })
        
        return {
            "status": "success",
            "segments": segments,
            "language": result["language"],
            "num_speakers": len(diarize_segments)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
    

def analyze_sentiment_from_text(transcript_text):
    """Analyze sentiment directly from text (without file)."""
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }
    
    speaker_texts = defaultdict(str)
    all_segments = []
    
    for line in transcript_text.split('\n'):
        match = re.match(r"\[\d+\.\d+ - \d+\.\d+\]\s+(Customer|Agent):\s+(.*)", line)
        if match:
            speaker = match.group(1)
            text = match.group(2).strip()
            speaker_texts[speaker] += " " + text
            all_segments.append(text)
    
    # Speaker-wise sentiment
    sentiment_results = {}
    for speaker in sorted(speaker_texts.keys()):
        try:
            sentiment = sentiment_pipe(speaker_texts[speaker][:512])[0]  # Limit to 512 tokens
            sentiment_results[speaker] = {
                "label": label_map[sentiment["label"]],
                "score": sentiment["score"]
            }
        except Exception as e:
            print(f"Error analyzing sentiment for {speaker}: {e}")
            sentiment_results[speaker] = {
                "label": "Error",
                "score": 0.0
            }
    
    # Overall sentiment
    if all_segments:
        try:
            full_text = " ".join(all_segments)[:512]
            overall_sentiment = sentiment_pipe(full_text)[0]
            sentiment_results["Overall"] = {
                "label": label_map[overall_sentiment["label"]],
                "score": overall_sentiment["score"]
            }
        except Exception as e:
            print(f"Error analyzing overall sentiment: {e}")
            sentiment_results["Overall"] = {
                "label": "Error",
                "score": 0.0
            }
    
    return sentiment_results

def analyze_tonal(audio_path, transcript_text):
    """Analyze tonal emotion from audio for all segments."""
    tone_map = {
        "neu": "Neutral",
        "hap": "Neutral",
        "exc": "Neutral",
        "sur": "Neutral",
        "sad": "Negative",
        "ang": "Negative",
        "fea": "Negative",
        "dis": "Negative"
    }
    
    # Parse transcript from text
    data = []
    for line in transcript_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*([A-Za-z]+):\s*(.*)", line)
        if match:
            data.append({
                "start": float(match.group(1)),
                "end": float(match.group(2)),
                "speaker": match.group(3),
                "text": match.group(4).strip()
            })
    
    if not data:
        return {"error": "No valid transcript data found"}
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.columns = ['start', 'end', 'speaker', 'text']
    
    # Load and resample audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Process all segments
    speaker_emotions = {speaker: [] for speaker in df['speaker'].unique()}
    
    for idx, row in df.iterrows():
        try:
            start = row["start"]
            end = row["end"]
            speaker = row["speaker"]
            
            start_sample = int(start * 16000)
            end_sample = int(end * 16000)
            segment_waveform = waveform[:, start_sample:end_sample]
            
            if segment_waveform.shape[1] < 1000:
                continue
            
            inputs = feature_extractor(
                segment_waveform.squeeze(), 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            with torch.no_grad():
                logits = hubert_tonal_model(**inputs).logits
            predicted_class = torch.argmax(logits).item()
            label = hubert_tonal_model.config.id2label[predicted_class]
            mapped_label = tone_map.get(label, "Neutral")
            speaker_emotions[speaker].append(mapped_label)
        except Exception as e:
            print(f"Error processing segment {idx}: {e}")
            continue
    
    # Summarize results
    tonal_results = {}
    for speaker, emotions in speaker_emotions.items():
        tonal_results[speaker] = dict(Counter(emotions))
    
    overall_emotion = []
    for emotions in speaker_emotions.values():
        overall_emotion.extend(emotions)
    
    tonal_results["Overall"] = dict(Counter(overall_emotion))
    
    return tonal_results

def create_sentiment_plot(sentiment_results):
    """Create a bar chart of sentiment analysis results."""
    speakers = list(sentiment_results.keys())
    sentiments = [result['label'] for result in sentiment_results.values()]
    scores = [result['score'] for result in sentiment_results.values()]
    
    color_map = {"Positive": "#28a745", "Neutral": "#ffc107", "Negative": "#dc3545", "Error": "#6c757d"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(speakers, scores, color=[color_map.get(s, "#007bff") for s in sentiments])
    
    ax.set_ylabel('Sentiment Score')
    ax.set_title('Sentiment Analysis Results')
    ax.set_ylim(0, 1.1)
    
    for bar, sentiment in zip(bars, sentiments):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, sentiment, 
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def create_tonal_plot(tonal_results):
    """Create a stacked bar chart of tonal analysis results."""
    if isinstance(tonal_results, dict) and "error" in tonal_results:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, tonal_results["error"], ha='center')
        return fig
    
    speakers = [k for k in tonal_results.keys() if k != "Overall"]
    emotions = ["Neutral", "Negative"]
    
    # Prepare data
    data = {e: [] for e in emotions}
    for speaker in speakers:
        counts = tonal_results.get(speaker, {})
        total = sum(counts.values()) or 1  # Avoid division by zero
        for e in emotions:
            data[e].append(counts.get(e, 0) / total)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0] * len(speakers)
    
    for idx, (emotion, values) in enumerate(data.items()):
        color = "#ffc107" if emotion == "Neutral" else "#dc3545"
        ax.bar(speakers, values, bottom=bottom, label=emotion, color=color)
        bottom = [sum(x) for x in zip(bottom, values)]
    
    ax.set_ylabel('Proportion')
    ax.set_title('Tonal Emotion Distribution')
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def retrieve_context(query, top_k=5):
    """Retrieve relevant context from rulebook."""
    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in docs])

def evaluate_call(call_text):
    """Evaluate call quality using LLM with more robust JSON handling."""
    # Simplified prompt that's more likely to produce valid JSON
    prompt = f"""
    Analyze this customer service call and provide evaluation metrics in JSON format:
    {{
      "Resolution": <1-10 score for problem resolution>,
      "Compliance": <1-10 score for protocol compliance>,
      "Satisfaction": <1-10 score for customer satisfaction>,
      "Final_rating": <average of the 3 scores>,
      "Evaluation": "<brief qualitative assessment>"
    }}

    Call Transcript:
    {call_text[:2000]}  # Limit length to avoid token limits

    Rulebook Context:
    {retrieve_context("call center best practices", top_k=2)}

    Respond ONLY with the JSON object, nothing else.
    """
    
    try:
        # Generate response with clearer instructions
        response = llm(prompt, max_new_tokens=200)[0]['generated_text']
        
        # More robust JSON extraction
        try:
            # Try direct parse first
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback to extraction if wrapped in text
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                return json.loads(match.group())
            return {"error": "Could not extract valid JSON from response"}
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {"error": f"Evaluation failed: {str(e)}"}

def process_audio(file_path, progress=gr.Progress()):
    """Full pipeline with progress updates and immediate outputs."""
    # Convert to WAV if needed
    progress(0.1, desc="Converting audio format...")
    wav_path = convert_to_wav(file_path)
    
    # Step 1: Transcribe with diarization
    progress(0.2, desc="Transcribing audio...")
    transcript = transcribe_with_diarization(wav_path)
    
    # Create classified transcript text
    transcript_text = "\n".join(
        f"[{utterance.start/1000:.2f} - {utterance.end/1000:.2f}] {utterance.speaker}: {utterance.text}"
        for utterance in transcript.utterances
    )
    
    # Step 2: Classify speakers
    progress(0.4, desc="Classifying speakers...")
    classified_transcript = classify_speakers(transcript_text)
    
    # Immediately yield the transcript
    yield classified_transcript, None, None, None
    
    # Step 3: Analyze sentiment (fast operation)
    progress(0.6, desc="Analyzing sentiment...")
    sentiment_results = analyze_sentiment_from_text(classified_transcript)
    sentiment_plot = create_sentiment_plot(sentiment_results)
    
    # Yield with sentiment results
    yield classified_transcript, sentiment_plot, None, None
    
    # Step 4: Analyze tonal emotion (slower operation)
    progress(0.7, desc="Analyzing tone...")
    tonal_results = analyze_tonal(wav_path, classified_transcript)
    tonal_plot = create_tonal_plot(tonal_results)
    
    # Yield with tonal results
    yield classified_transcript, sentiment_plot, tonal_plot, None
    
    # Step 5: Evaluate call quality
    progress(0.9, desc="Evaluating call quality...")
    evaluation_results = evaluate_call(classified_transcript)
    
    # Final yield with all results
    yield classified_transcript, sentiment_plot, tonal_plot, evaluation_results

# === GRADIO INTERFACE ===
with gr.Blocks() as demo:
    gr.Markdown("# 🎧 Advanced Call Center QA System")
    gr.Markdown("Upload a call recording to analyze agent performance with transcription, sentiment analysis, and quality evaluation.")
    
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="🎙️ Upload Call Recording")
        analyze_button = gr.Button("🚀 Analyze Call", variant="primary")
    
    with gr.Tabs():
        with gr.TabItem("📄 Transcription & Diarization"):
            transcription_output = gr.Textbox(label="Transcription and Diarization along with the Classification of Agent and Customer", lines=15)
        
        with gr.TabItem("📊 Sentiment Analysis"):
            sentiment_plot = gr.Plot(label="Sentiment Analysis Results")
        
        with gr.TabItem("🎭 Tonal Analysis"):
            tonal_plot = gr.Plot(label="Tonal Emotion Distribution")
        
        with gr.TabItem("🧠 Quality Evaluation"):
            evaluation_output = gr.JSON(label="Evaluation Results")
    
    analyze_button.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[transcription_output, sentiment_plot, tonal_plot, evaluation_output]
    )

if __name__ == "__main__":
    demo.launch()





