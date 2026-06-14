# app.py
import streamlit as st
from src.baselines import generate_lead3
from src.models import (
    get_target_device,
    load_t5_architecture,
    generate_abstractive_summary,
)


# --- Model Caching ---
@st.cache_resource
def load_all_pipelines():
    """Caches all networks into memory to prevent reloading on button clicks."""
    device = get_target_device()

    # Load Zero-Shot Model
    tok_zero, mod_zero = load_t5_architecture("t5-small", device)

    # Load Fine-Tuned Model
    tok_fine, mod_fine = load_t5_architecture(
        "./saved_model/final_t5_finetuned", device
    )

    return device, tok_zero, mod_zero, tok_fine, mod_fine


device, tokenizer_zero, model_zero, tokenizer_fine, model_fine = load_all_pipelines()

# --- UI Layout ---
st.set_page_config(page_title="News Summarizer AI", layout="wide")
st.title("📰 News Article Summarization System")
st.markdown(
    "Compare extractive baseline heuristics against zero-shot and fine-tuned Transformer models."
)

article_input = st.text_area(
    "Original News Article:", height=250, placeholder="Paste your article here..."
)

if st.button("🚀 Generate Summaries", type="primary"):
    if not article_input.strip():
        st.warning("Please paste an article first!")
    else:
        with st.spinner("Executing models..."):
            # Utilizing the new modular imports from src/
            lead3_out = generate_lead3(article_input, bullet_points=True)
            zero_out = generate_abstractive_summary(
                article_input, tokenizer_zero, model_zero, device, bullet_points=True
            )
            fine_out = generate_abstractive_summary(
                article_input, tokenizer_fine, model_fine, device, bullet_points=True
            )

            st.divider()
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("1️⃣ Lead-3 Baseline")
                st.info(lead3_out)
            with col2:
                st.subheader("2️⃣ T5 Zero-Shot")
                st.warning(zero_out)
            with col3:
                st.subheader("3️⃣ T5 Fine-Tuned")
                st.success(fine_out)
