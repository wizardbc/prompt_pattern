import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai
from ast import literal_eval

# load data
df_csv = pd.read_csv("./data/2302.11382v1_embeddings.csv", index_col=0).fillna('')
df_csv["embedding"] = df_csv.embedding.apply(literal_eval).apply(np.array)

with open('./data/toc.txt', 'r') as f:
  toc = f.read()

# search tools
def search_from_section_names(query:str) -> str:
  """Retrieve the LaTeX chunks of the paper named 'A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT' using the section, subsection and subsubsection names.
  The input is a list of three strings of the form `[section, subsection, subsubsection]`. Only the exact matchs of the names will be returned.

  Args:
    query: a list of three strings of the form `[section, subsection, subsubsection]`
  """
  query = list(query)
  df = df_csv.copy()
  if len(query) <= 3:
    query = query + ['']*(3-len(query))
  return df[
    (df['section'] == query[0])
    & (df['subsection'] == query[1])
    & (df['subsubsection'] == query[2])
  ][['section', 'subsection', 'subsubsection', 'text']].to_json()

def search_from_text(query:str, top_n:int=5, s:float=.0):
  """Retrieve the LaTeX chunks of the paper named 'A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT' using cosine similarity of text.
  The input is the user's query about the contents of the paper.

  Args:
    query: the user's query string.
  """
  df = df_csv.copy()
  query_embedding = np.array(genai.embed_content(
    model="models/text-embedding-004",
    content=query,
    task_type="retrieval_query",
  )["embedding"])
  df["similarity"] = df.embedding.apply(lambda x: np.dot(x, query_embedding))
  return df[df.similarity >= s].sort_values("similarity", ascending=False).head(top_n)[['text', 'similarity']].to_json()

# stream wrapper
def gemini_stream_text(response):
  for chunk in response:
    if parts:=chunk.parts:
      if text:=parts[0].text:
        yield text

# kwargs to markdown
def kwargs2mkdn(_indent:int=0, **kwargs):
  chunk = [' '*_indent + f"- {k}: {v}" for k, v in kwargs.items()]
  return '\n'.join(chunk)


st.title("💬 Chatbot with Paper")
st.caption("🚀 A Prompt Pattern Catalog to Enhance Prompt Engineering with Gemini 1.5 Flash")
st.write("https://arxiv.org/abs/2302.11382")

# Google API key
if "api_key" not in st.session_state:
  try:
    st.session_state.api_key = st.secrets["GOOGLE_API_KEY"]
  except:
    st.session_state.api_key = ''
    st.warning("Your Google API Key is not provided in `.streamlit/secrets.toml`, but you can input one in the sidebar for temporary use.")

# Sidebar for parameters
with st.sidebar:
  # Google API Key
  if st.session_state.api_key:
    genai.configure(api_key=st.session_state.api_key)
  else:
    st.session_state.api_key = st.text_input("Google API Key", type="password")

  st.subheader("Visible")
  system_checkbox = st.checkbox("system", value=False)
  f_call_checkbox = st.checkbox("tool", value=False)

  # ChatCompletion parameters
  st.header("Gemini Parameters")
  model_name = st.selectbox("model", ["gemini-1.5-flash", "gemini-1.5-pro"])
  generation_config = {
    "temperature": st.slider("temperature", min_value=0.0, max_value=1.0, value=1.0),
    "top_p": st.slider("top_p", min_value=0.0, max_value=1.0, value=0.95),
    "top_k": st.number_input("top_k", min_value=1, value=64),
    "max_output_tokens": st.number_input("max_output_tokens", min_value=1, value=8192),
  }
  
  # retriever parameters
  # st.header("Retriever Parameters")
  # n_chunks = st.number_input("number of chunks", min_value=1, value=5)

safety_settings={
  'harassment':'block_none',
  'hate':'block_none',
  'sex':'block_none',
  'danger':'block_none'
}

system_instruction=f"""You are an experienced prompt engineer.
You can retrieve the contents of the paper titled 'A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT'.

If you are not sure, then just say you don't know; never make up a story.
When you use the function `search_from_section_names`, first, you try fill all the three `[section, subsection, subsubsection]` names to get one or two chunks.
If you think we need more chunks, then ask the user want to get more.
If you cannot determine which section or (sub)subsection should be chosen, use the function `search_from_text` which is using cosine similarity of the query and the document body text.

You have to use Korean (한국어) if the user asks in Korean (한국어).
Otherwise you must use English.

Table of Contents:\n{toc}"""

if "chat_session" in st.session_state:
  chat_session = st.session_state.chat_session
else:
  model = genai.GenerativeModel(
    model_name=model_name,
    generation_config=generation_config,
    safety_settings=safety_settings,
    system_instruction=system_instruction,
    tools=[search_from_section_names, search_from_text],
  )
  chat_session = model.start_chat(enable_automatic_function_calling=True)
  st.session_state.chat_session = chat_session
  # with st.spinner("Generating..."):
  #   response = chat_session.send_message("Give me a simplified ToC containing all the section and subsection names, not subsubsection.")
  # st.rerun()

with st.sidebar:
  # chat controls
  st.header("Chat Control")
  # chat_role = st.selectbox("role", ["system", "model", "user", "tool"], index=1)
  col1, col2 = st.columns(2)
  with col1:
    st.button("Rewind", on_click=chat_session.rewind, use_container_width=True, type='primary')
  with col2:
    st.button("Clear", on_click=chat_session._history.clear, use_container_width=True)

# Display messages in history
if system_checkbox:
  with st.chat_message("system"):
    st.write(system_instruction)

# Display messages in history
for content in st.session_state.chat_session.history:
  full_text = ''
  for part in content.parts:
    if text:=part.text:
      full_text += text
    if f_call_checkbox:
      if fc:=part.function_call:
        full_text += f"Function Call\n- name: {fc.name}\n- args\n{kwargs2mkdn(4, **fc.args)}"
      if fr:=part.function_response:
        full_text += f"Function Response\n- name: {fr.name}\n- response\n{kwargs2mkdn(4, **fr.response)}"
  if full_text:
    with st.chat_message('human' if content.role == 'user' else 'ai'):
      st.write(full_text)

# Chat input
if prompt := st.chat_input("What is up?"):
  with st.chat_message('human'):
    st.write(prompt)
  with st.chat_message('ai'):
    with st.spinner("Generating..."):
      # response = chat_session.send_message(prompt, stream=True)
      response = chat_session.send_message(prompt)
    # st.write_stream(gemini_stream_text(response))
    st.write(response.parts[0].text)
    if f_call_checkbox:
      st.rerun()

