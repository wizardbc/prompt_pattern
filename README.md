# ChatGPT Playground
Put your Google API Key in `.streamlit/secrets.toml`.
```config
GOOGLE_API_KEY = "YOUR_API_KEY"
```
Run streamlit app.
```bash
pip install -r requirements.txt
streamlit run chatbot.py
```
Dependency: `google-generativeai`, `streamlit`.