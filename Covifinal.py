"""
Lung Health Diagnostic System - Offline Tkinter GUI
Author: Generated for Priyanka
Run: python lung_diagnostic_app.py

Assumptions:
- Keras/TensorFlow model saved as 'model.h5' in same folder.
- Model input size: 224x224 RGB (change IMG_SIZE below if needed).
- Class names order: ['Normal', 'Pneumonia', 'COVID'] (modify if your model uses different order)
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import tensorflow as tf

# ------------------------- USER CONFIG -------------------------
MODEL_PATH = "pneumoniaCovidTesterMy.keras"          # Change if your model filename differs
IMG_SIZE = (256, 256)            # Change to the size your model was trained with
CLASS_NAMES = ["Covid-19","Normal", "Pneumonia"]  # Change if your class order is different
CONFIDENCE_DISPLAY_DECIMALS = 1  # Number of decimals for percentages
# ----------------------- END USER CONFIG -----------------------

# Map class to suggestion text (customize as needed)
SUGGESTIONS = {
    'Normal': (
        "No sign of pneumonia/COVID detected with this model.\n"
        "If symptoms persist, consult a physician for clinical tests."
    ),
    'Pneumonia': (
        "Findings suggest pneumonia. Please consult a pulmonologist or physician.\n"
        "Consider chest CT and a clinical evaluation."
    ),
    'COVID-19': (
        "Findings suggest COVID-19 pneumonia. Isolate and consult a healthcare provider.\n"
        "Recommend PCR/rapid antigen testing and clinical assessment."
    ),
}

# ------------------------- Helper Functions -------------------------
def load_model(path):
    """Load the Keras model with error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    # load with compile=False for faster load if you don't need training
    model = tf.keras.models.load_model(path, compile=False)
    return model

def preprocess_image(pil_image):
    """
    Preprocess PIL image to model input.
    - convert to RGB
    - resize to IMG_SIZE
    - scale to [0,1]
    - expand dims
    If your model expects different preprocessing (e.g., imagenet), change here.
    """
    img = pil_image.convert('RGB')
    img = ImageOps.fit(img, IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)
    return arr

def predict_image(model, pil_image):
    """
    Returns (predicted_class, confidence_percent, all_probs_dict)
    """
    x = preprocess_image(pil_image)
    preds = model.predict(x)[0]  # shape (num_classes,)
    # if model outputs logits, softmax it
    if preds.sum() <= 1.0 + 1e-6 and preds.min() >= -1e-6:
        # might already be probabilities
        probs = preds
    else:
        probs = tf.nn.softmax(preds).numpy()
    idx = int(np.argmax(probs))
    cls = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
    conf = float(probs[idx]) * 100.0
    # Build dict of class->prob
    probs_dict = {CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}": float(probs[i])*100 for i in range(len(probs))}
    return cls, conf, probs_dict

# ------------------------- GUI Class -------------------------
class LungDiagnosticApp(tk.Tk):
    def __init__(self, model_path):
        super().__init__()
        self.title("COVID/Pneumonia/Normal Lung Health Diagnostic System")
        self.geometry("1000x700")
        self.resizable(False, False)
        # Styles
        self.style = ttk.Style(self)
        default_font = ("Poppins", 10)
        self.option_add("*Font", default_font)
        self.configure(bg="#f0f4f7")

        # Top frame: title and model status
        top = ttk.Frame(self)
        top.pack(fill="x", padx=16, pady=(12,6))
        title_label = ttk.Label(top, text="Covid/Pneumonia/Normal Lung Health Diagnostic System", font=("Century Gothic", 18, "bold"))
        title_label.pack(side="left")
        self.status_var = tk.StringVar(value="Loading model...")
        status_label = ttk.Label(top, textvariable=self.status_var, anchor="e")
        status_label.pack(side="right")

        # Main frame with left (image) and right (results)
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=16, pady=8)

        # Left panel - Image preview and upload button
        left = ttk.Frame(main, width=500, height=480)
        left.pack(side="left", fill="both", expand=True)
        left.pack_propagate(False)

        self.canvas = tk.Canvas(left, bg="white", bd=1, relief="sunken", width=480, height=400)
        self.canvas.pack(pady=(12,8))
        self.placeholder_text_id = self.canvas.create_text(240, 200, text="No image selected\nUpload a chest X-ray image", fill="#777", font=("Segoe UI", 14), justify="center")

        btn_frame = ttk.Frame(left)
        btn_frame.pack(pady=(6,0))
        upload_btn = ttk.Button(btn_frame, text="Upload Chest X-ray", command=self.on_upload)
        upload_btn.grid(row=0, column=0, padx=6)
        clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_preview)
        clear_btn.grid(row=0, column=1, padx=6)

        # Right panel - results and suggestions
        right = ttk.Frame(main, width=360)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        result_card = ttk.LabelFrame(right, text="Diagnosis")
        result_card.pack(fill="x", padx=8, pady=12)

        self.result_label = ttk.Label(result_card, text="No result yet", font=("Segoe UI", 12, "bold"))
        self.result_label.pack(anchor="w", padx=12, pady=(12,4))

        self.confidence_label = ttk.Label(result_card, text="")
        self.confidence_label.pack(anchor="w", padx=12)

        # Probabilities table
        probs_frame = ttk.Frame(result_card)
        probs_frame.pack(fill="x", padx=12, pady=(8,12))
        ttk.Label(probs_frame, text="Class").grid(row=0, column=0, sticky="w")
        ttk.Label(probs_frame, text="Confidence (%)").grid(row=0, column=1, sticky="e")

        self.prob_rows = {}
        for i, cname in enumerate(CLASS_NAMES, start=1):
            lbl = ttk.Label(probs_frame, text=cname)
            lbl.grid(row=i, column=0, sticky="w", pady=2)
            val = ttk.Label(probs_frame, text="-")
            val.grid(row=i, column=1, sticky="e")
            self.prob_rows[cname] = val

        # Suggestion card
        suggest_card = ttk.LabelFrame(right, text="Suggested Action")
        suggest_card.pack(fill="both", padx=8, pady=12, expand=True)
        self.suggestion_text = tk.Text(suggest_card, height=7, wrap="word", padx=8, pady=8, font=("Segoe UI", 10))
        self.suggestion_text.pack(fill="both", expand=True)
        self.suggestion_text.configure(state="disabled")

        # Footer: model info and credits
        footer = ttk.Frame(self)
        footer.pack(fill="x", padx=16, pady=(4,12))
        ttk.Label(footer, text=f"Model: {os.path.basename(model_path)} | Input Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}").pack(side="left")
        ttk.Label(footer, text="Offline • For educational use only").pack(side="right")

        # Internal state
        self.model = None
        self.current_image = None  # PIL image
        self.current_tkimage = None

        # Load model (synchronously so app is ready)
        try:
            self.update()  # show window first
            self.status_var.set("Loading model...")
            self.model = load_model(model_path)
            self.status_var.set("Model loaded ✅")
        except Exception as e:
            self.status_var.set("Model load failed ❌")
            messagebox.showerror("Model load error", f"Failed to load model:\n{e}\n\nMake sure '{MODEL_PATH}' exists and is a valid Keras model.")
            # Keep running UI, but disable upload
            self.model = None

    # ---------------- GUI Actions ----------------
    def on_upload(self):
        if self.model is None:
            messagebox.showwarning("Model not loaded", "Model not loaded. Cannot predict.")
            return
        filetypes = [("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select chest X-ray image", filetypes=filetypes)
        if not path:
            return
        try:
            pil_img = Image.open(path)
        except Exception as e:
            messagebox.showerror("File error", f"Could not open image:\n{e}")
            return
        self.current_image = pil_img
        self.show_preview(pil_img)
        # Run prediction in a short-lived thread so UI stays responsive
        threading.Thread(target=self._predict_and_show, daemon=True).start()

    def show_preview(self, pil_img):
        # Resize preview to fit canvas while preserving aspect ratio
        canvas_w = int(self.canvas['width'])
        canvas_h = int(self.canvas['height'])
        preview = ImageOps.contain(pil_img.convert('RGB'), (canvas_w-4, canvas_h-4))
        self.current_tkimage = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, image=self.current_tkimage)
        # small caption
        self.canvas.create_text(10, canvas_h-10, anchor="sw", text=f"{preview.width}x{preview.height}", fill="#333", font=("Segoe UI", 9))

    def clear_preview(self):
        self.current_image = None
        self.current_tkimage = None
        self.canvas.delete("all")
        self.placeholder_text_id = self.canvas.create_text(240, 200, text="No image selected\nUpload a chest X-ray image", fill="#777", font=("Segoe UI", 14), justify="center")
        self.result_label.config(text="No result yet")
        self.confidence_label.config(text="")
        for v in self.prob_rows.values():
            v.config(text="-")
        self.suggestion_text.configure(state="normal")
        self.suggestion_text.delete("1.0", "end")
        self.suggestion_text.configure(state="disabled")

    def _predict_and_show(self):
        # Called in background thread
        self.status_var.set("Predicting...")
        try:
            cls, conf, probs = predict_image(self.model, self.current_image)
        except Exception as e:
            self.status_var.set("Prediction failed")
            messagebox.showerror("Prediction error", f"An error occurred during prediction:\n{e}")
            return
        # Update UI on main thread
        self.after(0, lambda: self._update_results_ui(cls, conf, probs))
        self.status_var.set("Ready")

    def _update_results_ui(self, cls, conf, probs):
        conf_text = f"{conf:.{CONFIDENCE_DISPLAY_DECIMALS}f}% confidence"
        self.result_label.config(text=f"{cls} detected" if cls else "Result: -")
        self.confidence_label.config(text=conf_text)
        for cname, lbl in self.prob_rows.items():
            val = probs.get(cname, 0.0)
            lbl.config(text=f"{val:.{CONFIDENCE_DISPLAY_DECIMALS}f}")
        # Suggestion textbox
        suggestion = SUGGESTIONS.get(cls, "No suggestion available. Consult a healthcare professional.")
        # Optionally include a caution about model use
        suggestion_full = f"{suggestion}\n\nNote: This tool is for educational/demonstration use only and should not replace clinical assessment."
        self.suggestion_text.configure(state="normal")
        self.suggestion_text.delete("1.0", "end")
        self.suggestion_text.insert("1.0", suggestion_full)
        self.suggestion_text.configure(state="disabled")

# ------------------------- Main -------------------------
def main():
    root = LungDiagnosticApp(MODEL_PATH)
    root.mainloop()

if __name__ == "__main__":
    main()