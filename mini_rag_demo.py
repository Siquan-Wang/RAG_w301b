"""
最小化 RAG 演示
不需要 PDF，直接使用文本进行测试
"""
from openai import OpenAI
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

print("="*70)
print("🚀 最小化 RAG 系统演示")
print("="*70)

# 配置
INDEX_NAME = "mini_rag_demo"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 初始化
print("\n[1/6] 初始化组件...")
es = Elasticsearch(['http://localhost:9200'], verify_certs=False)
client = OpenAI(api_key=OPENAI_API_KEY)

# 使用轻量级嵌入模型
print("加载嵌入模型（首次会下载，约 120MB）...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("✓ 组件初始化完成")

# 创建索引
print("\n[2/6] 创建索引...")
try:
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
except:
    pass

mapping = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}
es.indices.create(index=INDEX_NAME, **mapping)
print(f"✓ 索引创建成功: {INDEX_NAME}")

# 准备知识库文档
print("\n[3/6] 准备知识库...")
documents = [
    "RAG（检索增强生成）是一种结合信息检索和文本生成的技术。它先从知识库检索相关文档，然后基于这些文档生成答案。",
    "Elasticsearch 是一个强大的搜索引擎，支持全文搜索和向量搜索。在 RAG 系统中，它用于存储和检索文档。",
    "向量嵌入（Vector Embedding）将文本转换为数字向量，使计算机能够理解文本的语义含义。相似的文本会有相似的向量。",
    "GPT 是 OpenAI 开发的大型语言模型，能够理解和生成自然语言。在 RAG 中，GPT 负责根据检索到的文档生成最终答案。",
    "混合搜索结合了关键词搜索和向量搜索的优势。关键词搜索擅长精确匹配，向量搜索擅长语义理解。",
]

# 生成嵌入并索引
print("生成嵌入向量并索引文档...")
for i, doc in enumerate(documents):
    embedding = model.encode(doc).tolist()
    es.index(
        index=INDEX_NAME,
        id=str(i),
        body={
            "text": doc,
            "embedding": embedding
        }
    )
es.indices.refresh(index=INDEX_NAME)
print(f"✓ 已索引 {len(documents)} 个文档")

# 查询函数
def rag_query(question: str) -> str:
    """RAG 查询流程"""
    # 1. 生成查询向量
    query_embedding = model.encode(question).tolist()
    
    # 2. 向量搜索
    search_result = es.search(
        index=INDEX_NAME,
        body={
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": 3,
                "num_candidates": 10
            },
            "_source": ["text"]
        }
    )
    
    # 3. 提取检索到的文档
    retrieved_docs = []
    for hit in search_result['hits']['hits']:
        retrieved_docs.append({
            'text': hit['_source']['text'],
            'score': hit['_score']
        })
    
    if not retrieved_docs:
        return "未找到相关信息"
    
    # 4. 构建上下文
    context = "\n\n".join([f"[文档{i+1}] {doc['text']}" for i, doc in enumerate(retrieved_docs)])
    
    # 5. 使用 GPT 生成答案
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "你是一个helpful的AI助手。请基于提供的文档回答问题，并在答案中标注引用来源（如[文档1]）。"
            },
            {
                "role": "user",
                "content": f"问题: {question}\n\n参考文档:\n{context}\n\n请回答:"
            }
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    return answer, retrieved_docs

# 演示查询
print("\n[4/6] 系统就绪！")
print("="*70)
print("现在可以提问了！\n")

# 示例问题
example_questions = [
    "什么是 RAG？",
    "Elasticsearch 的作用是什么？",
    "什么是向量嵌入？"
]

print("示例问题:")
for i, q in enumerate(example_questions, 1):
    print(f"  {i}. {q}")

print("\n" + "="*70)
print("输入你的问题（输入 'exit' 退出）:")
print("="*70 + "\n")

# 交互式问答
while True:
    try:
        question = input("💬 你的问题: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("\n再见！")
            break
        
        print("\n🔍 正在检索和生成答案...\n")
        
        answer, docs = rag_query(question)
        
        print("="*70)
        print("📝 AI 回答:")
        print("="*70)
        print(answer)
        
        print("\n" + "="*70)
        print(f"📚 检索到的文档 (共{len(docs)}个):")
        print("="*70)
        for i, doc in enumerate(docs, 1):
            print(f"\n[文档{i}] (相似度: {doc['score']:.4f})")
            print(doc['text'][:150] + "...")
        
        print("\n" + "="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n再见！")
        break
    except Exception as e:
        print(f"\n❌ 错误: {e}\n")

# 清理
print("\n[5/6] 清理...")
cleanup = input("是否删除演示索引？(y/n): ").strip().lower()
if cleanup == 'y':
    es.indices.delete(index=INDEX_NAME)
    print(f"✓ 已删除索引: {INDEX_NAME}")

print("\n[6/6] 完成！")
print("="*70)
print("🎉 RAG 系统演示结束！")
print("="*70)

