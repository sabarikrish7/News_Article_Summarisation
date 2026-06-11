```markdown
# News Article Summarization System

An end-to-end NLP pipeline implementing both extractive and abstractive summarization approaches on the CNN/DailyMail dataset. Built incrementally matching standard ML lifecycle workflows, complete with a comparative web dashboard.

## 📁 Project Structure

* `explore_data.py`: Dataset downloading, validation, and layout inspection.
* `evaluate_baseline.py`: Heuristic Lead-3 extractive baseline and ROUGE evaluation setup.
* `baseline_extractive.py`: Graph-based statistical text ranking baseline using spaCy and PyTextRank.
* `transformer_zeroshot.py`: Zero-shot abstractive evaluation using the pre-trained `t5-small` model.
* `train.py`: Fine-tuning script utilizing Hugging Face `Seq2SeqTrainer` on 5,000 samples.
* `evaluate_final.py`: Evaluation of the fine-tuned local weights with inference hyperparameter controls.
* `app.py`: Streamlit comparative web application displaying side-by-side summarization models.

---

## 🛠️ Installation & Setup

1. **Initialize Environment(with Python 3.11):**
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate

```

1. **Install Core Dependencies:**

```bash
uv pip install transformers datasets evaluate rouge_score torch pandas nltk spacy pytextrank accelerate sentencepiece protobuf streamlit torchvision
uv run python -m spacy download en_core_web_sm

```

---

## 🚀 Execution Guide

Run the scripts sequentially to reproduce the experiment benchmarks or launch the interface:

```bash
uv run explore_data.py          # 1. Inspect data structure
uv run evaluate_baseline.py     # 2. Run Lead-3 benchmark
uv run baseline_extractive.py   # 3. Run spaCy TextRank benchmark
uv run transformer_zeroshot.py  # 4. Run out-of-the-box T5 engine
uv run train.py                 # 5. Fine-tune model weights locally
uv run evaluate_final.py        # 6. Evaluate fine-tuned parameters
uv run streamlit run app.py     # 7. Launch the side-by-side web dashboard

```

---

## 📊 Evaluation Matrix (100 Test Samples)

| Approach | Model / Heuristic | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-LSUM |
| --- | --- | --- | --- | --- | --- |
| **Extractive** | TextRank (spaCy) | 26.23% | 8.46% | 17.24% | 23.08% |
| **Extractive** | Lead-3 Baseline | 30.60% | 12.73% | 21.56% | 27.47% |
| **Abstractive** | T5-Small (Zero-Shot) | 32.97% | 13.19% | 24.47% | 29.97% |
| **Abstractive** | T5-Small (Fine-Tuned) | **33.14%** | **13.49%** | 24.41% | **30.06%** |

---

## 💡 Key Findings

1. **The Lead-3 Anchor:** Due to the inverted-pyramid structure of news writing, traditional positional baselines (Lead-3) present a very strong lexical baseline that outscored more complex statistical graph methods like TextRank.
2. **Lexical vs. Semantic Scoring:** ROUGE scores rely strictly on exact word overlaps. Abstractive models that substitute valid synonyms are frequently penalized mathematically by ROUGE metrics, even when human readability is preserved or improved.
3. **Data Volume Scale:** Slicing training to a 5,000-sample subset provides a localized weight adjustment. While fine-tuning squeezed out improvements in targeted recall areas (`ROUGE1`, `ROUGE2`, `ROUGELSUM`), scaling to 50,000+ samples combined with gradient accumulation steps is required for dramatic performance shifts over a strong pre-trained baseline.
4. **Inference Hyperparameters:** Incorporating a strict `repetition_penalty=2.5` during the fine-tuned model generation step successfully checked text-looping artifacts, allowing the abstractive model to maintain stable linguistic fluency.
5. **Comparative Dashboard UI:** The Streamlit interface provides clear visual proof of training quality, demonstrating that custom fine-tuning successfully adapts the model to master proper news capitalization and structure which out-of-the-box zero-shot models lack.

```

```
