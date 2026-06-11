import streamlit as st
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --- Setup & Model Caching ---
# Caching prevents Streamlit from reloading heavy models on every UI interaction
@st.cache_resource
def setup_nltk():
    nltk.download("punkt", quiet=True)


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Zero-Shot Model
    tokenizer_zero = AutoTokenizer.from_pretrained("t5-small")
    model_zero = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(device)

    # 2. Load Fine-Tuned Model
    model_path = "./saved_model/final_t5_finetuned"
    tokenizer_fine = AutoTokenizer.from_pretrained(model_path)
    model_fine = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    return device, tokenizer_zero, model_zero, tokenizer_fine, model_fine


setup_nltk()
device, tokenizer_zero, model_zero, tokenizer_fine, model_fine = load_models()


# --- Summarization Logic ---
def generate_lead3(text):
    """Extracts the first 3 sentences."""
    sentences = nltk.sent_tokenize(text.strip())
    # Format nicely with bullet points
    return "\n\n".join([f"• {s}" for s in sentences[:3]])


def generate_transformer_summary(text, tokenizer, model):
    """Generates abstractive summary using specified model."""
    input_text = "summarize: " + text
    inputs = tokenizer(
        input_text, return_tensors="pt", max_length=512, truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=130,
            min_length=30,
            length_penalty=2.0,
            repetition_penalty=2.5,
            num_beams=4,
            early_stopping=True,
        )

    raw_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sentences = nltk.sent_tokenize(raw_summary)
    return "\n\n".join([f"• {s.strip()}" for s in sentences])


# --- UI Layout ---
# Set page to wide mode to fit all 3 columns nicely
st.set_page_config(page_title="News Summarizer AI", layout="wide")

st.title("📰 News Article Summarization System")
st.markdown(
    "Paste a news article below to compare extractive baseline heuristics against zero-shot and fine-tuned Transformer models."
)

# Input Area
article_input = st.text_area(
    "Original News Article:",
    height=250,
    placeholder="Paste your lengthy news article here...",
)

# Action Button
if st.button("🚀 Generate Summaries", type="primary"):
    if not article_input.strip():
        st.warning("Please paste an article first!")
    else:
        with st.spinner("Executing models... This might take a few seconds."):
            # Execute all three pipelines
            lead3_out = generate_lead3(article_input)
            zero_out = generate_transformer_summary(
                article_input, tokenizer_zero, model_zero
            )
            fine_out = generate_transformer_summary(
                article_input, tokenizer_fine, model_fine
            )

            # Display Results in a side-by-side 3-column layout
            st.divider()
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("1️⃣ Lead-3 Baseline")
                st.caption("Extractive: Blindly takes the first 3 sentences.")
                st.info(lead3_out)

            with col2:
                st.subheader("2️⃣ T5 Zero-Shot")
                st.caption("Abstractive: Pre-trained T5 out-of-the-box.")
                st.warning(zero_out)

            with col3:
                st.subheader("3️⃣ T5 Fine-Tuned")
                st.caption("Abstractive: Custom trained on CNN/DailyMail.")
                st.success(fine_out)
