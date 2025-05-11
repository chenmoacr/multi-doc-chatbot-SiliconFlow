import os
import sys
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter # 确保导入这个


# --- 从 OpenAI 替换为硅基流动 (SiliconFlow) ---
# from langchain_openai import ChatOpenAI # 旧的 OpenAI 聊天模型
# from langchain_openai import OpenAIEmbeddings # 旧的 OpenAI 嵌入模型

from SiliconFlowEmbeddings import SiliconFlowEmbeddings # 导入我们自定义的硅基流动嵌入类
from SiliconFlowChatModel import SiliconFlowChatModel   # 导入我们自定义的硅基流动聊天模型
# --- 替换结束 ---

# 加载环境变量 (如果 .env 文件中有其他配置，例如 Chroma 的路径等，或者未来API Key由此管理)
# 注意：目前 SiliconFlowEmbeddings 和 SiliconFlowChatModel 类内部是硬编码API Key的
# 如果您修改了这些类以从环境变量读取API Key，那么 .env 文件和 load_dotenv() 会更有用。
load_dotenv('.env')

# --- 提醒：API 密钥配置 ---
# 请确保您已在 SiliconFlowEmbeddings.py 和 SiliconFlowChatModel.py 文件中
# 正确设置了 SILICONFLOW_API_KEY。
# --- 提醒结束 ---

documents = []
# 从 ./docs 文件夹加载所有文件创建文档列表
print("正在加载文档...")
for file in os.listdir("docs"):
    file_path = os.path.join("docs", file)
    try:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"  已加载 PDF: {file}")
        elif file.endswith('.docx') or file.endswith('.doc'):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
            print(f"  已加载 DOCX/DOC: {file}")
        elif file.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8') # 明确指定编码，避免潜在问题
            documents.extend(loader.load())
            print(f"  已加载 TXT: {file}")
    except Exception as e:
        print(f"加载文件 {file} 时出错: {e}")
        # 可以选择跳过此文件或采取其他错误处理措施
        continue
print("文档加载完成。")

if not documents:
    print("错误：在 'docs' 文件夹中没有找到可加载的文档，或者所有文档加载失败。程序将退出。")
    sys.exit()

# 将文档分割成更小的块
print("正在分割文档...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=240,  # 大幅减小字符块大小
    chunk_overlap=40, # 相应的重叠
    separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""] # 可以保留或调整分隔符
)

documents = text_splitter.split_documents(documents)
print(f"文档分割完成，共得到 {len(documents)} 个文档块。")

# 将文档块转换为嵌入向量并保存到向量存储中
print("正在创建文本嵌入并构建向量数据库...")
# 使用我们自定义的硅基流动嵌入模型
# SiliconFlowEmbeddings 类默认使用 "netease-youdao/bce-embedding-base_v1"
embedding_function = SiliconFlowEmbeddings(batch_size=2) # 例如，保持为2

# 注意：如果您的 SiliconFlowEmbeddings 类在初始化时出现问题（例如API密钥未设置），这里会报错。
# 确保 SILICONFLOW_API_KEY 已在 SiliconFlowEmbeddings.py 中正确配置。

# 定义 Chroma 数据库的持久化路径
persist_directory = "./data_siliconflow" # 使用新的目录以区别于可能存在的 OpenAI 版本的数据
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

vectordb = Chroma.from_documents(
    documents,
    embedding=embedding_function,
    persist_directory=persist_directory
)
vectordb.persist() # 确保数据持久化
print(f"向量数据库已创建并持久化到 '{persist_directory}'。")

# 创建我们的问答链 (ConversationalRetrievalChain)
print("正在创建问答链...")
# 使用我们自定义的硅基流动聊天模型
# SiliconFlowChatModel 类默认使用 "deepseek-ai/DeepSeek-V3" 和 temperature=0.7
# 如果需要，可以显式传递参数，例如:
# llm = SiliconFlowChatModel(temperature=0.7, max_tokens=500)
llm = SiliconFlowChatModel(temperature=0.7) # 使用类中定义的默认值或按需调整

# 注意：如果您的 SiliconFlowChatModel 类在初始化时出现问题（例如API密钥未设置），这里会报错。
# 确保 SILICONFLOW_API_KEY 已在 SiliconFlowChatModel.py 中正确配置。

# 从向量数据库创建检索器
# search_kwargs={'k': 6} 表示检索6个最相关的文档块
retriever = vectordb.as_retriever(search_kwargs={'k': 6})

pdf_qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True, # 返回源文档块
    verbose=False # 设置为 True 可以看到链的详细运行过程
)
print("问答链创建成功。")

# 定义终端输出颜色
yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"
reset_color = "\033[0m" # 重置颜色

chat_history = []
print(f"\n{yellow}---------------------------------------------------------------------------------")
print('欢迎使用文档问答机器人 (硅基流动版)。您现在可以开始与您的文档进行交互了。')
print('输入 "exit", "quit", "q", 或 "f" 来退出程序。')
print('---------------------------------------------------------------------------------')

while True:
    try:
        query = input(f"{green}Prompt: {reset_color}")
    except UnicodeDecodeError:
        print(f"{yellow}检测到输入编码问题，请尝试使用标准字符输入。{reset_color}")
        continue
    except KeyboardInterrupt: # 处理 Ctrl+C
        print("\n检测到 Ctrl+C，正在退出...")
        sys.exit()


    if query.lower() in ["exit", "quit", "q", "f"]:
        print(f"{yellow}正在退出程序...{reset_color}")
        sys.exit()
    if query == '':
        continue

    try:
        # 调用问答链获取结果
        # ConversationalRetrievalChain 的 invoke 方法需要一个字典作为输入
        result = pdf_qa.invoke(
            {"question": query, "chat_history": chat_history}
        )
        print(f"\n{white}答案: " + result["answer"] + reset_color)

        # 如果需要显示源文档 (调试或验证时有用)
        # print("\n--- 源文档片段 ---")
        # for doc in result.get("source_documents", []):
        #     print(f"来源: {doc.metadata.get('source', '未知')}, 内容片段: {doc.page_content[:150]}...")
        # print("--- 源文档片段结束 ---\n")

        chat_history.append((query, result["answer"])) # 更新对话历史

    except ConnectionError as e:
        print(f"{yellow}网络连接错误: {e}。请检查您的网络连接和API服务状态。{reset_color}")
    except Exception as e:
        print(f"{yellow}处理您的问题时发生错误: {e}{reset_color}")
        # 在这里可以选择是否将错误信息加入聊天记录或进行其他处理
        # chat_history.append((query, f"发生错误: {e}")) # 可以选择不记录错误到历史，避免影响后续对话