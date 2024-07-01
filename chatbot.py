import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from ast import literal_eval
from typing import Literal, Union, Tuple

# load data
df_csv = pd.read_csv("./data/2302.11382v1_embeddings.csv", index_col=0).fillna('')
df_csv["embedding"] = df_csv.embedding.apply(literal_eval).apply(np.array)

with open("./data/toc.txt", 'r') as f:
  toc = f.read()

def get_embedding(text, model="text-embedding-3-small"):
  text = text.replace("\n", " ")
  return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_from_section_names(query: Tuple[str,str,str]) -> str:
  """Retrieve the matching (sub)sections of the paper named 'A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT'.
  """
  df = df_csv.copy()
  if len(query) <= 3:
    query = query + ['']*(3-len(query))
  return df[
    df['section'].str.contains(query[0])
    & df['subsection'].str.contains(query[1])
    & df['subsubsection'].str.contains(query[2])
  ][['section', 'subsection', 'subsubsection', 'text']].to_json()

def search_from_text(query:str, top_n:int=5, s:float=.0):
  """Retrieve the chunks of the paper named 'A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT'.
  """
  df = df_csv.copy()
  query_embedding = np.array(get_embedding(query, model="text-embedding-3-small"))
  df["similarity"] = df.embedding.apply(lambda x: np.dot(x, query_embedding))
  return df[df.similarity >= s].sort_values("similarity", ascending=False).head(top_n)[['text', 'similarity']].to_json()

retriever_tools=[
  {
    "type": "function",
    "function": {
      "name": "search_from_text",
      "description": "Retrieve the LaTeX chunks of the paper named 'A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT' using cosine similarity of text.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The user's query about the contents of the paper."
          },
        },
        "required": ["query"],
      },
    },
  },
  {
    "type": "function",
    "function": {
      "name": "search_from_section_names",
      "description": "Retrieve the LaTeX chunks of the paper named 'A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT' using the section, subsection and subsubsection names.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "A list of three strings of the form `[section, subsection, subsubsection]`. Only the exact matchs of the names will be returned. To select multiple chunks, `section`, `subsection` and `subsubsection` can be empty string, first, you try fill all the three elements to get one or two chunks."
          },
        },
        "required": ["query"],
      },
    },
  },
]

st.title("💬 Chatbot")
st.caption("🚀 A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT")
st.write("https://arxiv.org/abs/2302.11382")

# OpenAI API key
if "api_key" not in st.session_state:
  try:
    st.session_state.api_key = st.secrets["OPENAI_API_KEY"]
  except:
    st.session_state.api_key = ''
    st.warning("Your OpenAI API Key is not provided in `.streamlit/secrets.toml`, but you can input one in the sidebar for temporary use.")

# Initialize chat history
if "messages" not in st.session_state:
  st.session_state.messages = [
    {
      "role": "system",
      "content": f"""You are an experienced prompt engineer.
You can retrieve the contents of the paper titled 'A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT'.

If you are not sure, then just say you don't know; never make up a story.
When you use the function `search_from_section_names`, first, you try fill all the three `[section, subsection, subsubsection]` names to get one or two chunks.
If you think we need more chunks, then ask the user want to get more.

You have to use Korean (한국어) only if the user asks in Korean (한국어).
Otherwise you must use English.

Table of Contents (section, subsection, subsubsection):\n{toc}""",
    }
  ]

def undo():
  st.session_state.messages = st.session_state.messages[:-1]  

def clear():
  st.session_state.messages = st.session_state.messages[1:]

# Sidebar for parameters
with st.sidebar:
  # OpenAI API Key
  if st.session_state.api_key:
    client = OpenAI(api_key=st.session_state.api_key)
  else:
    st.session_state.api_key = st.text_input("OpenAI API Key", type="password")

  # Role selection and Undo
  st.header("Chat Control")
  chat_role = st.selectbox("role", ["system", "assistant", "user", "tool"], index=2)
  st.button("Undo", on_click=undo)

  st.subheader("Visible")
  system_checkbox = st.checkbox("system", value=False)
  f_call_checkbox = st.checkbox("tool", value=False)

  # ChatCompletion parameters
  st.header("ChatGPT Parameters")
  chat_params = {
    "model": st.selectbox("model", ["gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09", "gpt-4-0613", "gpt-3.5-turbo-0125"]),
    "temperature": st.slider("temperature", min_value=0.0, max_value=2.0, value=1.0),
    "max_tokens": st.number_input("max_tokens", min_value=1, value=2048),
    "top_p": st.slider("top_p", min_value=0.0, max_value=1.0, value=1.0),
    "presence_penalty": st.slider("presence_penalty", min_value=-2.0, max_value=2.0, value=0.0),
    "frequency_penalty": st.slider("frequency_penalty", min_value=-2.0, max_value=2.0, value=0.0),
    "stream": True,
  }

  # retriever parameters
  st.header("Retriever Parameters")
  n_chunks = st.number_input("number of chunks", min_value=1, value=5)

# Display messages in history
roles = ["user", "assistant"]
if system_checkbox:
  roles.append("system")
if f_call_checkbox:
  roles.append("tool")

for msg in st.session_state.messages:
  if (role := msg.get("role")) in roles:
    if content := msg.get("content", ""):
      with st.chat_message(role):
        if role == 'tool':
          st.json(content)
        else:
          st.write(content)
        
    if f_call_checkbox:
      if (tc:=msg.get("tool_calls")) is not None:
        tc = tc[0]
        if f_name := tc.get("function", {}).get("name", ""):
          f_args = tc.get("function", {}).get("arguments", "")
          with st.chat_message(role):
            st.write(f"- function name: {f_name}\n- args: {f_args}")

# In the case of the role of the last entry of the history is tool
if st.session_state.messages:
  if st.session_state.messages[-1].get("role") == "tool":
    with st.spinner("Generating..."):
      # ChatCompletion
      response = client.chat.completions.create(messages=st.session_state.messages, **chat_params)
      # Stream display
      with st.chat_message("assistant"):
        placeholder = st.empty()
      full_text = ""
      for chunk in response:
        if (text:=chunk.choices[0].delta.content) is not None:
          full_text += text
          placeholder.write(full_text + "▌")
      placeholder.write(full_text)
    st.session_state.messages.append({
      "role": "assistant",
      "content": full_text,
    })

# Chat input
if prompt := st.chat_input("What is up?"):
  # User message
  user_msg = {
    "role": chat_role,
    "content": prompt,
  }
  # tool role need name
  if chat_role == "tool":
    user_msg.update({"name": "dummy", "tool_call_id": "call_dummy"})
  # Display user message
  with st.chat_message(chat_role):
    st.write(prompt)
  # Append to history
  st.session_state.messages.append(user_msg)

  if chat_role == "user":
    # ChatCompletion
    response = client.chat.completions.create(
      messages=st.session_state.messages,
      tools=retriever_tools,
      **chat_params
    )
    # Stream display
    with st.spinner("Generating..."):
      with st.chat_message("assistant"):
        placeholder = st.empty()
      full_text = ""
      tool_calls = [{
        "index": 0,
        "type": "function",
        "function": {"arguments": ''},
      }]
      for chunk in response:
        delta = chunk.choices[0].delta
        if (text:=delta.content) is not None:
          full_text += text
          placeholder.write(full_text + "▌")
        if (tc:=delta.tool_calls) is not None:
          tc = tc[0]
          if tc.id:
            tool_calls[0]["id"] = tc.id
          if tc.function.name:
            tool_calls[0]["function"]["name"] = tc.function.name
          tool_calls[0]["function"]["arguments"] += tc.function.arguments if tc.function.arguments else ''
      placeholder.write(full_text)
    if full_text:
      st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
      })
    elif ftn_name:=tool_calls[0].get("function", {}).get("name", ''):
      st.session_state.messages.append({
        "role": "assistant",
        "tool_calls": tool_calls,
      })
      ftn_to_call = globals().get(ftn_name)
      with st.spinner("Retrieving `{ftn_name}`..."):
        st.session_state.messages.append({
          "role": "tool",
          "tool_call_id": tool_calls[0].get("id"),
          "name": ftn_name,
          "content": ftn_to_call(**literal_eval(tool_calls[0]["function"]["arguments"]), top_n=n_chunks) if ftn_name=="search_from_text" else ftn_to_call(**literal_eval(tool_calls[0]["function"]["arguments"])),
        })
      st.rerun()