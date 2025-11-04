\# RAG System - w301b Assignment



完整的 PDF RAG（检索增强生成）系统，支持文本、图像和表格处理。



\## 功能特性



\### 核心功能 ✅

\- ✅ \*\*Elasticsearch 集成\*\*: 本地部署，支持向量搜索

\- ✅ \*\*PDF 处理\*\*: 

&nbsp; - 文本提取和智能分块

&nbsp; - 图像提取和 AI 描述生成（GPT-4 Vision）

&nbsp; - 表格提取和处理

\- ✅ \*\*嵌入向量生成\*\*: 使用本地 Sentence Transformers 模型

\- ✅ \*\*向量搜索\*\*: 基于语义相似度的检索

\- ✅ \*\*问答生成\*\*: 基于检索内容生成带引用的答案



\## 技术栈



\- \*\*Python 3.12\*\*

\- \*\*Elasticsearch 8.11\*\* (Docker)

\- \*\*OpenAI GPT-4o-mini\*\* (文本生成和图像描述)

\- \*\*Sentence Transformers\*\* (本地嵌入模型)

\- \*\*PyMuPDF \& pdfplumber\*\* (PDF 处理)



\## 快速开始



\### 1. 安装依赖



pip install -r requirements.txt### 2. 启动 Elasticsearch



docker run -d --name elasticsearch -p 9200:9200 \\

&nbsp; -e "discovery.type=single-node" \\

&nbsp; -e "xpack.security.enabled=false" \\

&nbsp; docker.elastic.co/elasticsearch/elasticsearch:8.11.0### 3. 配置环境变量



创建 `.env` 文件：



