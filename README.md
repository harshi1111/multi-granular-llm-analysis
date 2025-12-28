# NeuraScan AI — Understand Why an LLM Answer Failed

**NeuraScan AI** is a web-based system that helps teams understand **where and why a Large Language Model (LLM) response starts to fail**, instead of only labeling it as right or wrong.

LLM answers often look fluent and confident while hiding:
- Logical inconsistencies
- Hallucinations
- Subtle factual errors

NeuraScan analyzes a **single LLM response across multiple layers and pinpoints the first layer where failure begins**, helping engineers debug and improve LLM behavior faster.

---
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Live Demo
> **See NeuraScan analyze real LLM responses**: 

https://github.com/user-attachments/assets/d12379dc-3ead-4a20-a1e8-defa7f7bc305






## 🏗️ Architecture Overview
<img width="770" height="506" alt="image" src="https://github.com/user-attachments/assets/8574901b-4b68-44e1-b8a4-a1c801383f41" />




## 🚀 What this project does

- **Token‑Level Analysis (Surface Fluency)**  
  Checks lexical stability, repetition patterns, and basic fluency at the word/subword level.

- **Sentence‑Level Analysis (Flow Between Sentences)**  
  Measures semantic coherence between sentences and detects obvious contradictions.

- **Reasoning‑Level Analysis (Logical Soundness)**  
  Extracts reasoning steps and looks for logical fallacies and inconsistent inferences.

- **Factual Consistency Check (Real‑World Accuracy)**  
  Uses a lightweight knowledge base to flag answers that are internally consistent but factually wrong  
  (e.g., “Australia is in Europe” → fluent but incorrect).

- **Cross‑Granular Comparison**  
  Combines all levels to show **where errors first appear** and how they propagate from local issues to high‑level reasoning failures.

- **Real LLM Integration (Groq API)**  
  Connects to Groq and runs analysis on live responses from models like `llama-3.3-70b` using free credits.

- **Interactive Web UI (Flask + JS)**  
  Paste a prompt, get the model’s answer, see scores, radar visualization, and a human‑readable error summary.

---

## 📊 How it works (high level)

1. **Query an LLM**  
   The app sends your prompt to a live model (via Groq API) and receives the response text.

2. **Decompose the response**  
   - Tokens (via tokenizer)  
   - Sentences (via spaCy)  
   - Reasoning steps (using pattern‑based extraction)

3. **Run level‑specific analyzers**  
   - Token analyzer → stability, repetition  
   - Sentence analyzer → coherence, contradictions  
   - Reasoning analyzer → step‑wise consistency and fallacies  
   - Factual checker → compare key claims with a small knowledge base

4. **Cross‑granular comparison**  
   A comparator module decides:
   - Which level first shows problems  
   - Whether errors propagate from lower to higher levels  
   - Whether the answer is “fluent but wrong” or “broken logic”, etc.

5. **Visualize & explain**  
   The Flask UI shows:
   - Radar chart (4 spokes: Fluency, Sentence Flow, Reasoning, Factuality)  
   - Numeric scores  
   - Short natural‑language explanation of what went wrong (or why the answer looks good).

---

## ✨ Key Features
- **Three-Level Analysis**: Token, sentence, and reasoning granularity inspection
- **Real LLM Integration**: Query Groq's Llama 3.3-70B with free credits
- **Error Propagation Tracking**: Visualize how issues spread across levels
- **Production Web Interface**: Real-time analysis with professional UI/UX
- **Dynamic Insights**: AI-powered patterns from analysis history
- **Full History System**: Manage and review past analyses

---
## 📈 Real-World Use Cases
- **Academic Research**: Study LLM hallucination patterns across granularity levels
- **Content Verification**: Check quality of AI-generated articles, reports, code
- **Education Tool**: Teach students critical analysis of AI outputs
- **LLM Development**: Debug and improve LLM applications by identifying failure points
- **Quality Assurance**: Automated checking of chatbot/assistant responses

## 🔧 Technical Stack
| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.9+, Flask 3.0, REST API, Background Workers |
| **NLP Pipeline** | spaCy, Sentence Transformers, Hugging Face Transformers |
| **Logic Validation** | SymPy (formal logic), NLI models |
| **Frontend** | HTML5, CSS3, JavaScript, Chart.js |
| **APIs** | Groq Cloud API (Llama 3.3-70B) |
| **Performance** | Threading, Task Queues, Response Chunking |

## 📝 Project Structure
```
multi_granular_llm_analysis/
├── app.py # Main Flask application
├── src/ # Core analysis modules
│ ├── token_analyzer.py
│ ├── sentence_analyzer.py
│ ├── reasoning_analyzer.py
│ ├── semantic_enhancer.py
│ └── cross_granular_comparator.py
├── config/ # Configuration files
├── templates/ # HTML templates
├── static/ # CSS, JS, assets
├── results/ # Analysis outputs
└── requirements.txt # Dependencies
```

## 🛠️ Installation
```bash
# Clone the repository
git clone <repository-url>
cd multi_granular_llm_analysis

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.template .env
# Edit .env and add your Groq API key

## ▶️ Running the web app

From the project root
python app.py
```

Then open `http://localhost:5000` in your browser.

1. Type a prompt (e.g., “Assume Australia is in Europe. Explain its climate.”)  
2. Click **Analyze**  
3. See:
   - The model’s raw answer  
   - Multi‑level scores  
   - Factual warnings if the answer contradicts the knowledge base  
   - A radar plot showing how good/bad the answer is at each level.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- [Groq](https://groq.com/) for providing free API access
- [Hugging Face](https://huggingface.co/) for transformer models
- [spaCy](https://spacy.io/) for NLP processing
   





