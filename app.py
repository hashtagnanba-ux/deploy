import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# --------------------------
# Load Model & Tokenizer
# --------------------------
MODEL_PATH = "model/fake_or_real_bert"  # Path to your saved model folder

@st.cache_resource
def load_model():
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

tokenizer, model = load_model()

# --------------------------
# Web App UI
# --------------------------
st.set_page_config(page_title="Fake or Real Text Detector", layout="centered")
st.title("🕵️ Fake or Real: Text Impostor Hunt")
st.markdown("""
This tool detects whether a given text is **Real** or **Fake**  
based on a fine-tuned BERT model from the Kaggle competition.
""")

# --------------------------
# User Input
# --------------------------
text_input = st.text_area("✏️ Enter the text to analyze:", height=200)

if st.button("🔍 Check Now"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to check.")
    else:
        try:
            # Tokenize the input text
            inputs = tokenizer(
                text_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_class = torch.argmax(probs, dim=1).item()

            # Labels - adjust if your training label mapping is different
            labels = {0: "Real", 1: "Fake"}
            confidence = probs[0][pred_class].item() * 100

            # Display result
            st.success(f"**Prediction:** {labels[pred_class]}")
            st.info(f"**Confidence:** {confidence:.2f}%")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
