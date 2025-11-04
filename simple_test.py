"""
简单测试脚本 - 不依赖 Elasticsearch，只测试 OpenAI 连接
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

print("="*70)
print("RAG 系统简单测试")
print("="*70)

# 加载环境变量
load_dotenv()

# 测试 1: 检查配置
print("\n[1/2] 检查配置...")
api_key = os.getenv('OPENAI_API_KEY')
if api_key and api_key.strip():
    print(f"✓ OpenAI API Key 已配置 ({api_key[:10]}...)")
else:
    print("✗ OpenAI API Key 未配置")
    print("\n请编辑 .env 文件，填入你的 OPENAI_API_KEY")
    input("\n按回车退出...")
    exit(1)

# 测试 2: 测试 OpenAI 连接
print("\n[2/2] 测试 OpenAI API 连接...")
try:
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "请只回复'测试成功'三个字"}
        ],
        max_tokens=10
    )
    
    result = response.choices[0].message.content
    print(f"✓ OpenAI API 连接成功")
    print(f"  AI 回复: {result}")
    
except Exception as e:
    print(f"✗ OpenAI API 连接失败: {e}")
    print("\n可能的原因:")
    print("  1. API Key 不正确")
    print("  2. API Key 没有余额")
    print("  3. 网络连接问题")
    input("\n按回车退出...")
    exit(1)

# 成功
print("\n" + "="*70)
print("✅ 基础测试通过！")
print("="*70)
print("\n下一步:")
print("  1. 安装并启动 Elasticsearch")
print("  2. 运行完整测试")
input("\n按回车退出...")

