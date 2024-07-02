import numpy as np
import pandas as pd
from io import StringIO
from ast import literal_eval
import google.generativeai as genai
import streamlit as st

st.set_page_config(
    page_title="Chat with Paper",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get help': 'https://github.com/wizardbc/prompt_pattern',
        'Report a bug': "https://github.com/wizardbc/prompt_pattern/issues",
        'About': "# Chat with Paper\nMade by Byung Chun Kim\n\nhttps://github.com/wizardbc/prompt_pattern"
    }
)

# load data
df_csv = pd.read_csv("./data/2302.11382v1_embeddings.csv", index_col=0).fillna('')
df_csv["embedding"] = df_csv.embedding.apply(literal_eval).apply(np.array)

with open('./data/toc.txt', 'r') as f:
  toc = f.read()

# search tools
def search_from_section_names(query:list[str]) -> str:
  """Retrieves LaTeX chunks from the paper "A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT" using the [section, subsection, subsubsection] names.

Args:
    query: A python list of three strings in the format `[section, subsection, subsubsection]`. Only exact matches of the names and order, will be returned.
  """
  query = [name if name else '' for name in list(query)]
  query += ['']*(3-len(query))
  df = df_csv.copy()
  return df[
    (df['section'] == query[0])
    & (df['subsection'] == query[1])
    & (df['subsubsection'] == query[2])
  ][['section', 'subsection', 'subsubsection', 'text']].to_json()

def search_from_text(query:str, top_n:int=5, s:float=.0):
  """Retrieves LaTeX chunks from the paper "A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT" using cosine similarity of text.

Args:
  query: The user's query string.
  top_n: The number of chunks to retrieve. The default value is 5. Start at 3 and recommend increasing it if needed.
  """
  df = df_csv.copy()
  query_embedding = np.array(genai.embed_content(
    model="models/text-embedding-004",
    content=query,
    task_type="retrieval_query",
  )["embedding"])
  top_n = int(top_n)
  df["similarity"] = df.embedding.apply(lambda x: np.dot(x, query_embedding))
  return df[df.similarity >= s].sort_values("similarity", ascending=False).head(top_n)[['text', 'similarity']].to_json()

# stream wrapper
# def gemini_stream_text(response):
#   for chunk in response:
#     if parts:=chunk.parts:
#       if text:=parts[0].text:
#         yield text

# kwargs to markdown
def kwargs2mkdn(_indent:int=0, **kwargs):
  chunk = [' '*_indent + f"- `{k}`: {v}" for k, v in kwargs.items()]
  return '\n'.join(chunk)

col1, col2 = st.columns(2, vertical_alignment='bottom')

with col1:
  st.title("💬 Chat with Paper")
  st.caption(":books: \"A Prompt Pattern Catalog to Enhance Prompt Engineering\" with Gemini 1.5")
  st.write("https://arxiv.org/abs/2302.11382")
  with st.container(border=True):
    st.write("논문의 내용을 가져와 대답해 줍니다.")
    st.write("내용을 가져오기위해 두가지 함수를 사용합니다.\n- `search_from_section_names`: 질문에 대한 대답이 있을 것으로 예상되는 논문의 섹션이름을 추출하여 해당 섹션의 내용을 가져 옵니다.\n- `search_from_text`: 논문의 섹션내용을 나타내는 embedding과 질문의 embedding의 cosine similarity를 사용하여 내용을 가져 옵니다.\n\n언어모델이 어떤 함수를 사용할지 판단하며, 명시적으로 함수명을 언급하여 해당 함수를 사용하도록 지시 할 수도 있습니다.\n\n아래의 목차를 참고하여 문서와 대화 해 보세요.")
  st.markdown("**Table of Contents:**")
  with st.expander("Introduction"):
    st.write("본 논문은 대화형 대규모 언어 모델(LLM)의 응용 분야를 확장하기 위해 프롬프트 패턴을 도입합니다.")
  with st.expander("Comparing Software Patterns with Prompt Patterns"):
    st.write("본 섹션에서는 소프트웨어 패턴과 프롬프트 패턴을 비교하여 프롬프트 엔지니어링을 위한 프레임워크를 제시합니다.\n\nSub-sections:\n- Overview of Software Patterns\n- Overview of Prompt Patterns\n- Evaluating Means for Defining a Prompt Pattern's Structure and Ideas\n- A Way Forward: Fundamental Contextual Statements")
  with st.expander("A Catalog of Prompt Patterns for Conversational LLMs"):
    st.write("본 섹션에서는 대화형 LLM 상호작용 및 출력 생성을 위한 소프트웨어 작업 자동화를 위한 프롬프트 패턴 카탈로그를 제시합니다.\n\nSub-sections:\n- Summary of the Prompt Pattern Catalog\n- The Meta Language Creation Pattern\n- The Output Automater Pattern\n- The Flipped Interaction Pattern\n- The Persona Pattern\n- The Question Refinement Pattern\n- The Alternative Approaches Pattern\n- The Cognitive Verifier Pattern\n- The Fact Check List Pattern\n- The Template Pattern\n- The Infinite Generation Pattern\n- The Visualization Generator Pattern\n- The Game Play Pattern\n- The Reflection Pattern\n- The Refusal Breaker Pattern\n- The Context Manager Pattern\n- The Recipe Pattern\n\nEach subsection, except the first, has subsubsections:\n- Intent and Context\n- Motivation\n- Structure and Key Ideas\n- Example Implementation\n- Consequences")
  with st.expander("Related Work"):
    st.write("본 섹션에서는 소프트웨어 패턴과 프롬프트 설계에 대한 기존 연구 및 LLM, 특히 ChatGPT의 성능 평가를 다룹니다.")
  with st.expander("Concluding Remarks"):
    st.write("본 논문은 ChatGPT와 같은 대규모 언어 모델(LLM)을 위한 프롬프트 패턴 카탈로그를 문서화하고 적용하는 프레임워크를 제시하며, 프롬프트 패턴 설계를 개선하여 대화형 LLM을 위한 새로운 기능을 창출하는 연구를 장려하고자 합니다.")

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

Table of Contents (each depth means [section, subsection, subsubsection]):\n{toc}"""

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
  btn_col1, btn_col2 = st.columns(2)
  with btn_col1:
    st.button("Rewind", on_click=chat_session.rewind, use_container_width=True, type='primary')
  with btn_col2:
    st.button("Clear", on_click=chat_session._history.clear, use_container_width=True)

with col2:
  messages = st.container()
  # display system instruction
  if system_checkbox:
    with messages.chat_message("system"):
      st.write(system_instruction)

  # display messages in history
  for content in st.session_state.chat_session.history:
    for part in content.parts:
      if text:=part.text:
        with messages.chat_message('human' if content.role == 'user' else 'ai'):
          st.write(text)
      if f_call_checkbox:
        if fc:=part.function_call:
          with messages.chat_message('ai'):
            st.write(f"Function Call\n- name: {fc.name}\n- args\n{kwargs2mkdn(4, **fc.args)}")
        if fr:=part.function_response:
          with messages.chat_message('human'):
            st.write(f"Function Response\n- name: {fr.name}\n- response\n  - `result`")
            st.json(fr.response["result"])
      else:
        if fr:=part.function_response:
          with messages.chat_message('ai'):
            st.dataframe(pd.read_json(StringIO(fr.response["result"]))[['section', 'subsection', 'subsubsection']])

  # chat input
  if prompt := st.chat_input("Ask me anything...", disabled=False if st.session_state.api_key else True):
    with messages.chat_message('human'):
      st.write(prompt)
    with messages.chat_message('ai'):
      with st.spinner("Generating..."):
        # response = chat_session.send_message(prompt, stream=True)
        response = chat_session.send_message(prompt)
      # st.write_stream(gemini_stream_text(response))
    st.rerun()