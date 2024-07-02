# Chat with Paper
Chat with the paper "A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT".

- arXiv link: https://arxiv.org/abs/2302.11382

This Streamlit application demonstrates the basic use of RAG and prompt patterns.

---

Put your Google API Key in .streamlit/secrets.toml or provide it in the sidebar for temporary use.
```config
GOOGLE_API_KEY = "YOUR_API_KEY"
```

Run streamlit app.
```bash
pip install -r requirements.txt
streamlit run chatbot.py
```
Dependencies: `google-generativeai`, `streamlit`, `pandas`.