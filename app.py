
# 启动 streamlit run app.py
import os
import streamlit as st


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============ 页面设置 ============
st.set_page_config(page_title="Multi-PDF RAG QA", layout="wide")
st.title("多PDF智能文档问答系统")

# ============ DeepSeek API Key ============
os.environ["OPENAI_API_KEY"] = st.secrets.get("DEEPSEEK_API_KEY", "")
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# ============ 初始化 Session State ============
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============ Sidebar: 上传 PDF ============
st.sidebar.header("上传 PDF 文档")
files = st.sidebar.file_uploader("上传多个 PDF 文件", type="pdf", accept_multiple_files=True)

# ============ 构建知识库 ============
@st.cache_resource(show_spinner=False)
def build_vectorstore(files) -> FAISS:
    docs = []
    for file in files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(file.name)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

if files and st.sidebar.button("构建文档向量库"):
    with st.spinner("正在构建向量数据库..."):
        st.session_state.vectorstore = build_vectorstore(files)
    st.sidebar.success("文档库构建完成")

# ============ RAG 构建 ============
def get_rag_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template(
        """
你是一个基于文档的专业问答助手，请严格依据【上下文】内容进行回答。

【历史对话】
{history}

【上下文】
{context}

【问题】
{question}

请给出准确回答，并在最后列出引用的文档来源与页码。
"""
    )

    chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# ============ 聊天窗口 ============
st.subheader("文档问答")

query = st.text_input("请输入你的问题")

if st.button("提问"):
    if st.session_state.vectorstore is None:
        st.warning("请先上传并构建文档向量库")
    else:
        rag_chain = get_rag_chain(st.session_state.vectorstore)
        result = rag_chain.invoke({
            "question": query,
            "history": "\n".join(st.session_state.chat_history)
        })

        st.session_state.chat_history.append(f"用户：{query}")
        st.session_state.chat_history.append(f"助手：{result}")

# ============ 显示历史对话 ============
for msg in st.session_state.chat_history:
    if msg.startswith("用户"):
        st.markdown(f"**🧑 {msg}**")
    else:
        st.markdown(f"🤖 {msg}")
