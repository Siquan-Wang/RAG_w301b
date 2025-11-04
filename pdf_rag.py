"""
完整的 PDF RAG 系统
支持文本、图像和表格处理
"""
import os
import fitz  # PyMuPDF
import pdfplumber
from openai import OpenAI
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import base64
from datetime import datetime

load_dotenv()

# 配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)
es = Elasticsearch(['http://localhost:9200'], verify_certs=False)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

class PDFProcessor:
    """处理 PDF 文档"""
    
    def __init__(self, index_name):
        self.index_name = index_name
        self.setup_index()
    
    def setup_index(self):
        """创建索引"""
        try:
            if es.indices.exists(index=self.index_name):
                es.indices.delete(index=self.index_name)
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
                    },
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "content_type": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"}
                }
            }
        }
        es.indices.create(index=self.index_name, **mapping)
        print(f"✓ 索引创建成功: {self.index_name}")
    
    def extract_text(self, pdf_path):
        """提取文本"""
        print("\n[1/3] 提取文本...")
        chunks = []
        
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                # 简单分块（每500字符一块）
                words = text.split()
                chunk_size = 100  # 单词数
                
                for i in range(0, len(words), chunk_size):
                    chunk_text = ' '.join(words[i:i+chunk_size])
                    if chunk_text.strip():
                        chunks.append({
                            'text': chunk_text,
                            'source': os.path.basename(pdf_path),
                            'page': page_num + 1,
                            'content_type': 'text'
                        })
        
        doc.close()
        print(f"✓ 提取了 {len(chunks)} 个文本块")
        return chunks
    
    def extract_images(self, pdf_path):
        """提取图像并生成描述"""
        print("\n[2/3] 提取图像...")
        image_data = []
        
        doc = fitz.open(pdf_path)
        image_count = 0
        
        for page_num in range(min(len(doc), 5)):  # 只处理前5页以节省成本
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list[:2]):  # 每页最多2张图
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # 生成描述
                    caption = self.caption_image(image_bytes, page_num + 1)
                    
                    image_data.append({
                        'text': f"图像描述: {caption}",
                        'source': os.path.basename(pdf_path),
                        'page': page_num + 1,
                        'content_type': 'image'
                    })
                    image_count += 1
                except Exception as e:
                    print(f"  跳过图像 {img_index}: {e}")
        
        doc.close()
        print(f"✓ 处理了 {image_count} 张图像")
        return image_data
    
    def caption_image(self, image_bytes, page_num):
        """使用 GPT-4 Vision 生成图像描述"""
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "简要描述这张图片的内容（1-2句话）。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }],
                max_tokens=100
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"第 {page_num} 页的图像"
    
    def extract_tables(self, pdf_path):
        """提取表格"""
        print("\n[3/3] 提取表格...")
        table_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages[:5]):  # 只处理前5页
                    tables = page.extract_tables()
                    
                    for table in tables:
                        if table and len(table) > 1:
                            # 转换为文本
                            table_text = self.table_to_text(table)
                            
                            table_data.append({
                                'text': f"表格内容: {table_text}",
                                'source': os.path.basename(pdf_path),
                                'page': page_num + 1,
                                'content_type': 'table'
                            })
        except Exception as e:
            print(f"  表格提取失败: {e}")
        
        print(f"✓ 提取了 {len(table_data)} 个表格")
        return table_data
    
    def table_to_text(self, table):
        """将表格转换为文本"""
        lines = []
        for row in table[:5]:  # 只取前5行
            row_text = ' | '.join([str(cell or '') for cell in row])
            lines.append(row_text)
        return '\n'.join(lines)
    
    def index_documents(self, documents):
        """索引文档"""
        print(f"\n索引 {len(documents)} 个文档...")
        
        for i, doc in enumerate(documents):
            try:
                embedding = model.encode(doc['text']).tolist()
                
                doc_body = {
                    'text': doc['text'],
                    'embedding': embedding,
                    'source': doc['source'],
                    'page': doc['page'],
                    'content_type': doc['content_type'],
                    'chunk_id': f"{doc['source']}_p{doc['page']}_{i}"
                }
                
                es.index(index=self.index_name, id=f"doc_{i}", document=doc_body)
                
                if (i + 1) % 10 == 0:
                    print(f"  已索引 {i + 1}/{len(documents)}")
            except Exception as e:
                print(f"  索引失败 {i}: {e}")
        
        es.indices.refresh(index=self.index_name)
        print(f"✓ 索引完成")
    
    def process_pdf(self, pdf_path):
        """完整处理流程"""
        print(f"\n处理 PDF: {pdf_path}")
        print("="*70)
        
        # 提取所有内容
        text_chunks = self.extract_text(pdf_path)
        image_data = self.extract_images(pdf_path)
        table_data = self.extract_tables(pdf_path)
        
        # 合并所有文档
        all_docs = text_chunks + image_data + table_data
        
        # 索引
        self.index_documents(all_docs)
        
        print(f"\n处理完成!")
        print(f"  文本块: {len(text_chunks)}")
        print(f"  图像: {len(image_data)}")
        print(f"  表格: {len(table_data)}")
        print(f"  总计: {len(all_docs)} 个文档")
        
        return len(all_docs)


class RAGQuery:
    """RAG 查询"""
    
    def __init__(self, index_name):
        self.index_name = index_name
    
    def search(self, query, top_k=5):
        """搜索"""
        query_embedding = model.encode(query).tolist()
        
        result = es.search(
            index=self.index_name,
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": top_k,
                    "num_candidates": 50
                },
                "_source": ["text", "source", "page", "content_type"]
            }
        )
        
        docs = []
        for hit in result['hits']['hits']:
            docs.append({
                'text': hit['_source']['text'],
                'source': hit['_source']['source'],
                'page': hit['_source']['page'],
                'type': hit['_source']['content_type'],
                'score': hit['_score']
            })
        
        return docs
    
    def generate_answer(self, query):
        """生成答案"""
        # 检索
        docs = self.search(query)
        
        if not docs:
            return "未找到相关信息", []
        
        # 构建上下文
        context = "\n\n".join([
            f"[文档{i+1}] (来源: {doc['source']}, 第{doc['page']}页, 类型: {doc['type']})\n{doc['text'][:300]}"
            for i, doc in enumerate(docs)
        ])
        
        # 生成答案
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个helpful的AI助手。请基于提供的文档回答问题，并标注引用来源。如果文档中没有相关信息，请明确说明。"
                },
                {
                    "role": "user",
                    "content": f"问题: {query}\n\n参考文档:\n{context}\n\n请回答:"
                }
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        return answer, docs


def main():
    """主函数"""
    print("="*70)
    print("🚀 完整 PDF RAG 系统")
    print("="*70)
    
    # 检查 PDF 文件
    pdf_dir = "test_pdf"
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"\n⚠️  请将 PDF 文件放到 '{pdf_dir}' 目录中")
        input("\n按回车退出...")
        return
    
    print(f"\n找到 {len(pdf_files)} 个 PDF 文件:")
    for i, f in enumerate(pdf_files, 1):
        print(f"  {i}. {f}")
    
    # 选择文件
    if len(pdf_files) == 1:
        pdf_path = os.path.join(pdf_dir, pdf_files[0])
    else:
        choice = input(f"\n选择要处理的文件 (1-{len(pdf_files)}): ").strip()
        try:
            idx = int(choice) - 1
            pdf_path = os.path.join(pdf_dir, pdf_files[idx])
        except:
            print("无效选择")
            return
    
    index_name = "pdf_rag_index"
    
    # 处理 PDF
    processor = PDFProcessor(index_name)
    processor.process_pdf(pdf_path)
    
    # 交互式问答
    print("\n" + "="*70)
    print("📚 PDF 处理完成！现在可以提问了")
    print("="*70)
    print("输入问题（输入 'exit' 退出）\n")
    
    rag = RAGQuery(index_name)
    
    while True:
        try:
            question = input("💬 你的问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'q']:
                break
            
            print("\n🔍 正在检索和生成答案...\n")
            
            answer, docs = rag.generate_answer(question)
            
            print("="*70)
            print("📝 AI 回答:")
            print("="*70)
            print(answer)
            
            print("\n" + "="*70)
            print(f"📚 检索到的文档 (共{len(docs)}个):")
            print("="*70)
            for i, doc in enumerate(docs, 1):
                print(f"\n[文档{i}] {doc['source']} - 第{doc['page']}页 - {doc['type']}")
                print(f"相似度: {doc['score']:.4f}")
                print(doc['text'][:200] + "...")
            
            print("\n" + "="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}\n")
    
    # 清理
    cleanup = input("\n是否删除索引？(y/n): ").strip().lower()
    if cleanup == 'y':
        es.indices.delete(index=index_name)
        print(f"✓ 已删除索引: {index_name}")
    
    print("\n完成！")


if __name__ == "__main__":
    main()

