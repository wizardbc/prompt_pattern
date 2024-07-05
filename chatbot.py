import numpy as np
import pandas as pd
from io import StringIO
from ast import literal_eval
import google.generativeai as genai
import streamlit as st

### set_page
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

### load data
df_csv = pd.read_csv("./data/2302.11382v1_embeddings.csv", index_col=0).fillna('')
df_csv["embedding"] = df_csv.embedding.apply(literal_eval).apply(np.array)

with open('./data/toc.txt', 'r') as f:
  toc = f.read()

### search tools
def search_from_section_names(query:list[str]) -> str:
  """Retrieves LaTeX chunks from the paper "A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT" using the [section, subsection, subsubsection] names.

Args:
    query: A python list of three strings in the format `[section, subsection, subsubsection]`.
  """
  query = [name if name else '' for name in list(query)]
  query += ['']*(3-len(query))
  df = df_csv.copy()
  res_df = df[
    (df['section'] == query[0])
    & (df['subsection'] == query[1])
    & (df['subsubsection'] == query[2])
  ]
  if len(res_df)==0:
    res_df = df[
      df['section'].str.contains(query[0])
      & df['subsection'].str.contains(query[1])
      & df['subsubsection'].str.contains(query[2])
    ]
  return res_df[['section', 'subsection', 'subsubsection', 'text']].to_json()

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
  return df[df.similarity >= s].sort_values("similarity", ascending=False).head(top_n)[['section', 'subsection', 'subsubsection', 'text', 'similarity']].to_json()

tools = {
  'search_from_section_names': search_from_section_names,
  'search_from_text': search_from_text,
}

def ftn_codeblock(fn, args):
  res = "```python\n"
  res += str(fn)
  res += '(\n'
  for k,v in args.items():
    if isinstance(v, str):
      res += f'  {k}="{v}"\n'
    else:
      res += f'  {k}={v}\n'
  res += ')\n```'
  return res

### stream wrapper
### gemini does not provide the `automatic_function_calling` and stream output simultaneously.
def gemini_stream_text(response):
  for chunk in response:
    if parts:=chunk.parts:
      if text:=parts[0].text:
        yield text

@st.experimental_dialog("🚨 Error")
def error(err, msg=''):
  st.write(f"We've got error\n```python\n{err}\n```")
  if msg:
    st.text(msg)
  if st.button("Ok"):
    st.rerun()

### Google API key
if "api_key" not in st.session_state:
  try:
    st.session_state.api_key = st.secrets["GOOGLE_API_KEY"]
  except:
    st.session_state.api_key = ''

with st.sidebar:
  if st.session_state.api_key:
    genai.configure(api_key=st.session_state.api_key)
  else:
    st.session_state.api_key = st.text_input("Google API Key", type="password")

### Layout
with st.sidebar:
  st.header("Visibility")
  st.caption("Panel on Right")
  help_checkbox = st.checkbox("Help", value=False)
  toc_checkbox = st.checkbox("Table of Contents", value=True)
  memo_checkbox = st.checkbox("Memo", value=True)
  st.caption("Messages")
  f_call_checkbox = st.checkbox("Function Call", value=False)
  f_response_checkbox = st.checkbox("Function Response", value=False)

tab_main, tab_memo, tab_system = st.tabs(["Main", "Memo", "System Instruction"])

with tab_main:
  st.title("💬 Chat with Paper")
  st.caption(":books: \"A Prompt Pattern Catalog to Enhance Prompt Engineering\" with Gemini 1.5")
  st.write("https://arxiv.org/abs/2302.11382")
  st.divider()
  if not st.session_state.api_key:
    st.warning("Your Google API Key is not provided in `.streamlit/secrets.toml`, but you can input one in the sidebar for temporary use.", icon="⚠️")

  if help_checkbox or toc_checkbox or memo_checkbox:
    col_l, col_r = st.columns([6,4], vertical_alignment='bottom')
    with col_l:
      messages = st.container()
  else:
    messages = st.container()

# memo
if "memo" not in st.session_state:
  st.session_state.memo = []

with tab_memo:
  st.header("Memo", divider=True)
  for m in st.session_state.memo:
    st.write(m)
    st.divider()

_system_instruction = f"""You are a retrieval-augmented generative engine. 
Your primary task is to retrieve the contents of the paper titled "A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT".

**Retrieval Process:**

1. **Attempt Retrieval:** Always try to retrieve the paper's content first, even if you are confident in your knowledge.
2. **Retrieval Failure:** If you cannot find the paper, simply state that you are unable to retrieve it. **Do not** rely on your prior knowledge.
3. **Structured Retrieval:** When using the `search_from_section_names` function, prioritize filling at least one of parameters `[section, subsection, subsubsection]` using the table of contents to retrieve a relevant chunk. However, `section`, `subsection` or `subsubsection` can be empty strings (`''`) if necessary. But, all three cannot be empty strings.
4. **Cosine Similarity:** If you cannot determine the appropriate section or subsection, use the `search_from_text` function, which leverages cosine similarity between the query and the document body text. 
5. **Additional Retrieval:** If you believe more chunks are needed, ask the user if they would like to retrieve additional information.

**Language Handling:**

* Respond in Korean (한국어) if the user's query is in Korean.
* Respond in English otherwise.

**Table of Contents:**

{toc}"""

# system prompt
with tab_system:
  system_instruction = st.text_area("system instruction", _system_instruction, height=512)

# help, toc, memo in col_r
if help_checkbox:
  with col_r:
    with st.container(border=True):
      st.write("논문의 내용을 가져와 대답해 줍니다.")
      st.write("내용을 가져오기위해 두가지 함수를 사용합니다.\n- `search_from_section_names`: 질문에 대한 대답이 있을 것으로 예상되는 논문의 섹션이름을 추출하여 해당 섹션의 내용을 가져 옵니다.\n- `search_from_text`: 논문의 섹션내용을 나타내는 embedding과 질문의 embedding의 cosine similarity를 사용하여 내용을 가져 옵니다.\n\n언어모델이 어떤 함수를 사용할지 판단하며, 명시적으로 함수명을 언급하여 해당 함수를 사용하도록 지시 할 수도 있습니다.\n\n아래의 목차를 참고하여 문서와 대화 해 보세요.")
if toc_checkbox:
  with col_r:
    with st.container(border=True):
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
if memo_checkbox:
  with col_r, st.container(border=True):
    st.subheader("Memo", divider=True)
    len_memo_m_1 = len(st.session_state.memo) - 1
    for i, m in enumerate(st.session_state.memo):
      st.write(m)
      st.button("Remove", on_click=st.session_state.memo.remove, args=[m], key=f'_memo_{i}')
      if i < len_memo_m_1:
        st.divider()

### gemini parameters
with st.sidebar:
  st.header("Gemini Parameters")
  model_name = st.selectbox("model", ["gemini-1.5-flash", "gemini-1.5-pro"])
  generation_config = {
    "temperature": st.slider("temperature", min_value=0.0, max_value=1.0, value=1.0),
    "top_p": st.slider("top_p", min_value=0.0, max_value=1.0, value=0.95),
    "top_k": st.number_input("top_k", min_value=1, value=64),
    "max_output_tokens": st.number_input("max_output_tokens", min_value=1, value=8192),
  }

safety_settings={
  'harassment':'block_none',
  'hate':'block_none',
  'sex':'block_none',
  'danger':'block_none'
}

### gemini
if "history" not in st.session_state:
  st.session_state.history = []

model = genai.GenerativeModel(
  model_name=model_name,
  generation_config=generation_config,
  safety_settings=safety_settings,
  system_instruction=system_instruction,
  tools=tools.values(),
)
chat_session = model.start_chat(
  history=st.session_state.history,
  enable_automatic_function_calling=False
)

### chat controls
def rewind():
  if len(chat_session.history) >= 2:
    chat_session.rewind()
  if len(chat_session.history) >= 2:
    part = chat_session.history[-1].parts[0]
    if part.function_call:
      chat_session.rewind()
  st.session_state.history = chat_session.history

def clear():
  chat_session.history.clear()
  st.session_state.history = chat_session.history

with st.sidebar:
  st.header("Chat Control")
  btn_col1, btn_col2 = st.columns(2)
  with btn_col1:
    st.button("Rewind", on_click=rewind, use_container_width=True, type='primary')
  with btn_col2:
    st.button("Clear", on_click=clear, use_container_width=True)

### display messages in history
for i, content in enumerate(chat_session.history):
  for part in content.parts:
    if text:=part.text:
      with messages.chat_message('human' if content.role == 'user' else 'ai'):
        st.write(text)
        if content.role == 'model':
          st.button("Memo", on_click=st.session_state.memo.append, args=[text], key=f'_btn_{i}')
    if f_call_checkbox and (fc:=part.function_call):
      with messages.chat_message('ai'):
        st.write(f"**Function Call**:\n\n{ftn_codeblock(fc.name, fc.args)}")
    if f_response_checkbox and (fr:=part.function_response):
      with messages.chat_message('retriever', avatar="📜"):
        if "search_" in fr.name:
          retriever_df = pd.read_json(StringIO(fr.response["result"]))
          st.dataframe(retriever_df.loc[:, (retriever_df.columns != "text")])
          with st.expander("Content"):
            for text in retriever_df.text:
              st.text(text)
        else:
          st.write(f"Function Response\n- name: {fr.name}\n- response\n  - `result`")
          st.json(fr.response["result"])

### chat input
if prompt := st.chat_input("Ask me anything...", disabled=False if st.session_state.api_key else True):
  with messages.chat_message('human'):
    st.write(prompt)
  with messages.chat_message('ai'):
    with st.spinner("Generating..."):
      try:
        response = chat_session.send_message(prompt, stream=True)
        text = st.write_stream(gemini_stream_text(response))
        st.session_state.history = chat_session.history
      except genai.types.StopCandidateException as e:
        error(e, "구글의 contents filter 에 걸렸을 수 있습니다.\n저작권이 있는 문서를 생성하려는 경우 일 가능성이 있습니다.")
      except genai.types.BrokenResponseError as e:
        error(e, "구글의 contents filter 에 걸렸을 수 있습니다.\n저작권이 있는 문서를 생성하려는 경우 일 가능성이 있습니다.")
      # function response
      fr_parts = []
      for part in response.parts:
        if fc := part.function_call:
          st.toast(f"**Function Calling**\n`{fc.name}`")
          fr_parts.append(
            genai.protos.Part(
              function_response=genai.protos.FunctionResponse(
                name=fc.name,
                response={"result": tools[fc.name](**fc.args)}))
          )
      if fr_parts:
        try:
          response = chat_session.send_message(fr_parts)
          text = st.write_stream(gemini_stream_text(response))
          st.session_state.history = chat_session.history
          if f_call_checkbox or f_response_checkbox:
            st.rerun()
        except genai.types.StopCandidateException as e:
          st.session_state.history = st.session_state.history[:-2]
          error(e, "구글의 contents filter 에 걸렸을 수 있습니다.\n저작권이 있는 문서를 생성하려는 경우 일 가능성이 있습니다.")
        except genai.types.BrokenResponseError as e:
          st.session_state.history = st.session_state.history[:-2]
          error(e, "구글의 contents filter 에 걸렸을 수 있습니다.\n저작권이 있는 문서를 생성하려는 경우 일 가능성이 있습니다.")
    st.button("Memo", on_click=st.session_state.memo.append, args=[text], key=f'_btn_last')