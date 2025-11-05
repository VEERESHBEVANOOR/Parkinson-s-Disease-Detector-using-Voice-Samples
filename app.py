import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter, ImageDraw
import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
import joblib
import pyaudio
import wave
import threading
import random

# --- Phrases ---
PHRASES = [
    "The quick brown fox jumps over the lazy dog.",
    "We were away a year ago.",
    "A good book is a good friend.",
    "She sells seashells by the seashore.",
    "The birch canoe slid on the smooth planks.",
    "I owe you a yoyo today."
]

# --- Feature Extraction & Prediction ---
def extract_features(file_path):
    try:
        sound = parselmouth.Sound(file_path)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
        local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        mfcc = sound.to_mfcc(number_of_coefficients=12)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        features = {'local_jitter': local_jitter, 'rap_jitter': rap_jitter,
                    'local_shimmer': local_shimmer, 'hnr': hnr}
        for i in range(len(mfcc_mean)):
            features[f'mfcc_{i+1}_mean'] = mfcc_mean[i]
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def predict_parkinsons(audio_file_path):
    try:
        model = joblib.load('parkinsons_model.joblib')
        scaler = joblib.load('scaler.joblib')
    except FileNotFoundError:
        return "Error: Model/scaler not found.", "#FFA500"
    
    new_features = extract_features(audio_file_path)
    if new_features is None:
        return "Could not process audio file.", "#FF4C4C"
        
    training_cols = scaler.get_feature_names_out()
    features_df = pd.DataFrame([new_features], columns=training_cols)
    features_df.fillna(0, inplace=True)
    
    scaled_features = scaler.transform(features_df)
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)
    
    if prediction[0] == 1:
        result = f"Parkinson's Detected (Confidence: {prediction_proba[0][1]*100:.2f}%)"
        color = "#FF4C4C"
    else:
        result = f"Healthy (Confidence: {prediction_proba[0][0]*100:.2f}%)"
        color = "#4CFF4C"
    return result, color

# --- Silence Detection ---
def is_silent(file_path, threshold=500):
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            if np.max(np.abs(audio_data)) < threshold: return True
        return False
    except (wave.Error, ValueError):
        return True

# --- Recording Logic ---
is_recording = False
recording_thread = None

def reset_buttons():
    upload_button['state'] = 'normal'
    record_button['text'] = 'Record Live Audio'
    record_button['state'] = 'normal'

def analyze_from_file():
    if is_recording: return
    file_path = filedialog.askopenfilename(title="Select an Audio File", filetypes=(("WAV files", "*.wav"),))
    if not file_path: return
    canvas.itemconfig(status_text_id, text="Analyzing...")
    canvas.itemconfig(result_text_id, text="")
    root.update_idletasks()
    result, color = predict_parkinsons(file_path)
    canvas.itemconfig(status_text_id, text="Analysis Complete")
    canvas.itemconfig(result_text_id, text=result, fill=color)

def toggle_recording():
    global is_recording, recording_thread
    if not is_recording:
        is_recording = True
        upload_button['state'] = 'disabled'
        record_button['text'] = 'Stop Recording'
        chosen_phrase = random.choice(PHRASES)
        canvas.itemconfig(status_text_id, text="Please read aloud:")
        canvas.itemconfig(result_text_id, text=chosen_phrase, fill="white")
        recording_thread = threading.Thread(target=record_audio, daemon=True)
        recording_thread.start()
    else:
        is_recording = False
        record_button['state'] = 'disabled'
        canvas.itemconfig(status_text_id, text="Saving and processing...")

def record_audio():
    CHUNK, FORMAT, CHANNELS, RATE = 1024, pyaudio.paInt16, 1, 44100
    FILENAME = "temp_recording.wav"
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    frames = []
    silent_chunks = 0
    max_silent_chunks = int(RATE / CHUNK * 2)  # 2 seconds silence
    while is_recording:
        data = stream.read(CHUNK)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.max(np.abs(audio_data)) < 500:
            silent_chunks += 1
        else:
            silent_chunks = 0
        if silent_chunks > max_silent_chunks:
            break  # auto-stop if too long silence
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    if len(frames) == 0 or silent_chunks > max_silent_chunks:
        canvas.itemconfig(result_text_id, text="No audio detected. Please try again.", fill="#FFA500")
        canvas.itemconfig(status_text_id, text="Recording stopped.")
        reset_buttons()
        return
    
    with wave.open(FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    root.after(0, analyze_recorded_audio, FILENAME)

def analyze_recorded_audio(file_path):
    if is_silent(file_path):
        canvas.itemconfig(status_text_id, text="No sound detected.")
        canvas.itemconfig(result_text_id, text="Please try recording again.", fill="#FFA500")
    else:
        canvas.itemconfig(status_text_id, text="Analyzing...")
        result, color = predict_parkinsons(file_path)
        canvas.itemconfig(status_text_id, text="Analysis Complete")
        canvas.itemconfig(result_text_id, text=result, fill=color)
    reset_buttons()

# --- Main Window ---
root = tk.Tk()
root.title("Parkinson's Detector")
root.attributes('-fullscreen', True)
root.configure(bg="#000000")
screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
canvas = tk.Canvas(root, width=screen_width, height=screen_height, highlightthickness=0, bg="#000000")
canvas.pack(fill="both", expand=True)

# --- Background ---
try:
    bg_image_pil = Image.open("bg.png").resize((screen_width, screen_height), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image_pil)
    canvas.create_image(0, 0, image=bg_photo, anchor="nw")
except FileNotFoundError:
    canvas.config(bg="#000000")
    bg_image_pil = Image.new('RGB', (screen_width, screen_height), "#000000")

# --- Transparent and Visible Frosted Glass Button with Rounded Edges ---
def create_animated_glass_button(x, y, text, command, w=250, h=60):
    # Crop background for button area
    cropped = bg_image_pil.crop((max(0, x-w//2), max(0, y-h//2), min(screen_width, x+w//2), min(screen_height, y+h//2)))
    cropped = cropped.resize((w, h), Image.LANCZOS)
    
    # Normal state: transparent frosted glass with subtle white overlay for visibility
    blurred_normal = cropped.filter(ImageFilter.GaussianBlur(radius=5))
    overlay_normal = Image.new('RGBA', (w, h), (255, 255, 255, 30))  # Low opacity for transparency but visible
    blurred_normal.paste(overlay_normal, (0,0), overlay_normal)
    # Rounded white border for visibility
    draw = ImageDraw.Draw(blurred_normal)
    draw.rounded_rectangle([0, 0, w-1, h-1], outline=(255, 255, 255, 100), width=2, radius=15)
    
    # Hover state: slightly more opaque for better visibility on interaction
    blurred_hover = cropped.filter(ImageFilter.GaussianBlur(radius=8))
    overlay_hover = Image.new('RGBA', (w, h), (255, 255, 255, 50))  # Increased slightly for visibility
    blurred_hover.paste(overlay_hover, (0,0), overlay_hover)
    draw_hover = ImageDraw.Draw(blurred_hover)
    draw_hover.rounded_rectangle([0, 0, w-1, h-1], outline=(255, 255, 255, 150), width=3, radius=15)
    
    img_normal = ImageTk.PhotoImage(blurred_normal)
    img_hover = ImageTk.PhotoImage(blurred_hover)
    
    btn = tk.Label(root, image=img_normal, text=text, font=("Courier New", 18, "bold"),
                   compound="center", fg="#FFFFFF", cursor="hand2", bg="#000000")  # White text for high visibility
    btn.image = img_normal
    btn.img_hover = img_hover
    btn.img_normal = img_normal
    btn.place(x=x, y=y, anchor="center")
    
    def on_click(event):
        # Simple click feedback: brief brighter state
        btn.config(image=btn.img_hover)
        root.after(100, lambda: btn.config(image=btn.img_normal))
        command()
    
    btn.bind("<Button-1>", on_click)
    btn.bind("<Enter>", lambda e: btn.config(image=btn.img_hover))
    btn.bind("<Leave>", lambda e: btn.config(image=btn.img_normal))
    return btn

# --- UI Elements ---
x_anchor = screen_width * 0.75
# Transparent title (no background box)
title_id = canvas.create_text(x_anchor, screen_height * 0.25, text="Parkinson's Detector", font=("Courier New", 52, "bold"), fill="#FFFFFF")

status_text_id = canvas.create_text(x_anchor, screen_height * 0.45, text="Ready for analysis", font=("Courier New", 16, "italic"), fill="#CCCCCC")
result_text_id = canvas.create_text(x_anchor, screen_height * 0.55, text="", font=("Courier New", 26, "bold"), fill="#FFFFFF", width=screen_width*0.45, justify="center")

upload_button = create_animated_glass_button(x_anchor, screen_height*0.75, "Upload Audio File", analyze_from_file)
record_button = create_animated_glass_button(x_anchor, screen_height*0.82, "Record Live Audio", toggle_recording)
exit_button = create_animated_glass_button(screen_width-100, 50, "Exit", root.destroy, w=120, h=50)

root.mainloop()
