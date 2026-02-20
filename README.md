# 多PDF智能文档问答系统

## 项目简介：

针对科研论文与技术文档阅读效率低、信息检索成本高以及大模型易产生幻觉的问题，参考 GitHub 开源 RAG 项目（[Agentic RAG with LangGraph](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/rag_tutorials/ai_blog_search)）思路，设计并实现了一套基于检索增强生成（RAG）架构的多 PDF 智能文档问答系统。系统通过“文档解析–语义检索–大模型生成”的协同机制，实现对多篇 PDF 文档的高精度问答与多轮对话交互，并引入答案溯源机制增强结果可信性。

## 项目流程：

![工作流程](https://github.com/aliwa8168/Multi-PDF-Intelligent-Document-Q-A-System/blob/main/workflow.jpg?raw=true)

## 所用技术：

**编程语言：** Python 3.10+

**框架：** LangChain

**向量数据库：** FAISS

**模型：**

- 向量嵌入（Embeddings）：HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- 对话模型（Chat）：DeepSeek API（`deepseek-chat`）

**文档加载器：** LangChain `PyPDFLoader`

**文档切分器：** RecursiveCharacterTextSplitter

**向量存储构建：** LangChain FAISS VectorStore

**检索方式：** VectorStore Retriever（Top-K=4）

**Prompt 构建：** ChatPromptTemplate

**输出解析器：** StrOutputParser

**用户界面（UI）：** Streamlit

**多轮对话管理：** Streamlit Session State

## Requirements

1. **安装依赖**:

   ```
   pip install -r requirements.txt
   ```

   

2. **运行**:

   ```
   streamlit run app.py
   ```

3. **使用**：

   .streamlit下面的secrets.toml粘贴你的deep seek api key

   ```
   DEEPSEEK_API_KEY="your api key"
   ```


   
