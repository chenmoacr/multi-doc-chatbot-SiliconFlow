# 基于 LangChain 和硅基流动（SiliconFlow）的多文档问答聊天机器人

本项目是 [smaameri/multi-doc-chatbot](https://github.com/smaameri/multi-doc-chatbot) 的一个分支版本，经过修改以适配并使用硅基流动（SiliconFlow）提供的API服务，替代了原项目中对 OpenAI API 的依赖。

本项目旨在实现一个能够读取本地文件夹中的多个文档（包括 PDF, DOCX, TXT 格式）并基于这些文档内容与用户进行问答和聊天的机器人。机器人能够记忆对话历史，并利用 LangChain 框架进行文档处理、文本嵌入、向量存储、信息检索和对话生成。

## 主要特性

* **多文档读取**: 支持处理 `.pdf`, `.docx`, `.doc`, 和 `.txt` 多种格式的文档。
* **基于文档的问答**: 用户可以针对已加载文档的内容进行提问。
* **对话历史记忆**: 聊天机器人能够记住之前的对话上下文，进行连贯的交流。
* **硅基流动（SiliconFlow）集成**:
    * **文本嵌入 (Embeddings)**: 使用硅基流动提供的 `netease-youdao/bce-embedding-base_v1` 模型（0.5K 上下文窗口）。
    * **对话生成 (Chat Completions)**: 使用硅基流动提供的 `deepseek-ai/DeepSeek-V3` 模型。
* **本地向量存储**: 使用 ChromaDB 将文档的嵌入向量持久化存储在本地 (`./data_siliconflow` 目录)。
* **命令行交互**: 通过终端与聊天机器人进行交互。

## 核心技术栈

* Python 3.x
* LangChain
* 硅基流动 (SiliconFlow) API
* ChromaDB (向量数据库)
* Requests (HTTP请求)

## 项目结构
```plaintext
.
├── .venv/                  # Python 虚拟环境 (可选)
├── data_siliconflow/       # ChromaDB 向量数据库存储目录
├── docs/                   # 存放待处理的源文档 (PDF, DOCX, TXT)
├── img/                    # (原项目图片资源，可能未使用)
├── pycache/            # Python 缓存
├── .env.example            # 环境变量示例文件 (本项目API Key目前主要在代码内配置)
├── .gitignore
├── LICENSE                 # 项目许可证
├── multi-doc-chatbot.py    # 原版 OpenAI 项目主脚本 (参考用)
├── multi-doc-chatbot_SiliconFlow.py # 本项目主运行脚本 (硅基流动版)
├── README.md               # 本说明文件
├── requirements.txt        # Python 依赖包列表
├── SiliconFlowChatModel.py # 自定义 LangChain 聊天模型类 (对接硅基流动)
├── SiliconFlowEmbeddings.py # 自定义 LangChain 嵌入类 (对接硅基流动)
├── single-doc.py           # (原项目脚本)
└── single-long-doc.py      # (原项目脚本)
```

## 安装与配置

1.  **克隆仓库** (如果您已在本地，则跳过此步)
    ```bash
    git clone <您的仓库URL>
    cd <仓库目录名>
    ```

2.  **创建并激活 Python 虚拟环境** (推荐)
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```
    *确保 `requirements.txt` 文件包含了所有必要的库，例如 `langchain`, `langchain-community`, `langchain-core`, `requests`, `chromadb`, `pypdf`, `docx2txt`, `python-dotenv`, `tiktoken` 等。*

4.  **配置硅基流动 API 密钥 (非常重要!)**
    本项目目前将 API 密钥占位符硬编码在 `SiliconFlowEmbeddings.py` 和 `SiliconFlowChatModel.py` 文件中。您需要：
    * 打开 `SiliconFlowEmbeddings.py` 文件。
    * 找到以下行：
        ```python
        SILICONFLOW_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ```
    * **将其中的 `"sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"` 替换为您从硅基流动官方获取的真实 API 密钥。**
    * 对 `SiliconFlowChatModel.py` 文件执行相同的操作。

    *注意：直接在代码中硬编码 API 密钥仅建议用于个人测试和演示。在更正式的环境或开源分享时，推荐使用环境变量 (例如通过 `.env` 文件和 `python-dotenv` 库加载) 来管理敏感信息。如果需要，您可以自行修改代码以从环境变量读取密钥。*

## 使用方法

1.  **准备文档**:
    * 将您希望聊天机器人学习的文档（PDF, DOCX, DOC, TXT格式）放入项目根目录下的 `docs/` 文件夹中。
    * 如果 `docs/` 文件夹不存在，请创建它。

2.  **运行聊天机器人**:
    在激活虚拟环境并配置好 API 密钥后，从项目根目录运行主脚本：
    ```bash
    python multi-doc-chatbot_SiliconFlow.py
    ```

3.  **开始交互**:
    * 脚本启动后，会首先加载和处理 `docs/` 文件夹中的文档，创建或加载向量数据库。这个过程可能需要一些时间，具体取决于文档数量和大小。
    * 当看到欢迎信息和 `Prompt:` 提示符后，您就可以开始提问了。
    * 输入您的问题，然后按 Enter。
    * 若要退出程序，请输入 `exit`, `quit`, `q`, 或 `f`。

## 关键参数与定制化提示

* **文本块大小 (`chunk_size`)**: 在 `multi-doc-chatbot_SiliconFlow.py` 文件中，`RecursiveCharacterTextSplitter` 的 `chunk_size` 参数（当前建议为 `240` 字符左右）对于确保文本块不超过硅基流动 `netease-youdao/bce-embedding-base_v1` 模型的 `0.5K` (512 Token) 上下文限制至关重要。如果遇到与文本长度相关的错误，可能需要进一步调整此参数。
* **嵌入批处理大小 (`batch_size`)**: 在 `multi-doc-chatbot_SiliconFlow.py` 中实例化 `SiliconFlowEmbeddings` 时，可以传递 `batch_size` 参数（当前建议为 `2`）。此参数控制一次向嵌入API发送多少个文本块。如果处理大量文档时初始嵌入速度较慢，可以尝试适当增大此值（例如 `5`, `8`, `16`），但需注意不要超过API的整体请求负载限制。
* **模型名称**: 如果硅基流动将来更新了模型名称或您希望尝试其他兼容模型，可以在 `SiliconFlowEmbeddings.py` 和 `SiliconFlowChatModel.py` 类定义中修改 `model_name` 属性的默认值，或在实例化时传入新的模型名称。

## 许可证

本项目遵循 [LICENSE](./LICENSE) 文件中的许可协议。

## 致谢

* 感谢 [Saamer Mansoor (smaameri)](https://github.com/smaameri) 创建了优秀的原始项目 [multi-doc-chatbot](https://github.com/smaameri/multi-doc-chatbot)。
* 感谢硅基流动（SiliconFlow）提供的API服务。