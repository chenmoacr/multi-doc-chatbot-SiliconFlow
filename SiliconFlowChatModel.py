# SiliconFlowChatModel.py

import requests
import json # 主要用于调试时打印 payload
from typing import List, Optional, Any, Dict

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage, # 用于处理未明确映射的 role
)
from langchain_core.outputs import ChatGeneration, ChatResult

# 定义您的硅基流动 API 密钥 (请替换成您的真实密钥)
# 注意：直接在代码中硬编码 API 密钥仅适用于测试和演示环境。
# 在生产环境中，请使用更安全的方式管理密钥，例如环境变量。
SILICONFLOW_API_KEY = "sk-cqbudipmjjawthzlkygswcpeaexznqfivulrotjjpvsxlpnb" # <--- 请在这里替换成您的 API 密钥 (与 Embeddings 类使用相同的密钥)

class SiliconFlowChatModel(BaseChatModel):
    """
    一个与硅基流动 (SiliconFlow) 聊天服务交互的自定义 LangChain ChatModel 类。

    它默认使用 deepseek-ai/DeepSeek-V3 模型。
    """
    model_name: str = "deepseek-ai/DeepSeek-V3"
    api_key: str = SILICONFLOW_API_KEY
    api_url: str = "https://api.siliconflow.cn/v1/chat/completions"
    temperature: float = 0.7
    max_tokens: int = 512 # API 默认值，可调整
    top_p: float = 0.7 # API 默认值
    # 根据 API 文档，还有其他参数可以按需添加，例如:
    # stop: Optional[List[str]] = None
    # top_k: int = 50
    # frequency_penalty: float = 0.5
    # ...等

    # 用于存储会话的属性 (如果需要持久化会话或请求)
    # _session: requests.Session = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError("未提供硅基流动 API 密钥。请设置 SILICONFLOW_API_KEY 或在构造函数中传入 api_key。")
        if "sk-" not in self.api_key and len(self.api_key) < 50:
             print(f"警告: API 密钥 '{self.api_key[:10]}...' 可能格式不正确或不是一个有效的硅基流动API Key。请检查。")
        # self._session = requests.Session() # 可以考虑使用 Session

    @property
    def _llm_type(self) -> str:
        return "siliconflow-chat-model"

    def _convert_message_to_dict(self, message: BaseMessage) -> Dict[str, str]:
        """将 LangChain 的 BaseMessage 转换为硅基流动 API 所需的字典格式。"""
        if isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        elif isinstance(message, ChatMessage): # 处理通用 ChatMessage
            return {"role": message.role, "content": message.content}
        else:
            raise ValueError(f"不支持的消息类型: {type(message)}")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        核心方法，用于生成聊天回复。
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 将 LangChain messages 转换为 API 格式
        api_messages = [self._convert_message_to_dict(m) for m in messages]

        # 构造请求体 (payload)
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": api_messages,
            "stream": False, # 非流式输出
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            # 可以根据需要添加更多参数
        }
        if stop:
            payload["stop"] = stop
        
        # 如果 kwargs 中有其他 API 支持的参数，也可以添加到 payload 中
        # 例如: top_k, frequency_penalty 等，需要确保它们是 API 支持的
        # for key in ["top_k", "frequency_penalty", "response_format", "tools"]:
        #     if key in kwargs:
        #         payload[key] = kwargs[key]

        # print(f"发送到 API 的 Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}") # 调试用

        try:
            # response = self._session.post(self.api_url, json=payload, headers=headers) # 如果使用 session
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()  # 检查 HTTP 错误
            response_json = response.json()
            # print(f"从 API 收到的响应: {json.dumps(response_json, indent=2, ensure_ascii=False)}") # 调试用

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"调用硅基流动聊天 API 时发生网络错误: {e}") from e
        except Exception as e:
            raise RuntimeError(f"调用硅基流动聊天 API 时发生未知错误: {e}") from e


        if "choices" not in response_json or not response_json["choices"]:
            raise ValueError("API 响应格式错误：未找到 'choices' 或 'choices' 为空。")

        # 提取回复
        choice = response_json["choices"][0]
        if "message" not in choice or "content" not in choice["message"]:
            raise ValueError("API 响应格式错误：'choice' 中未找到 'message' 或 'content'。")

        assistant_message_content = choice["message"]["content"]
        
        # 如果 API 返回了 'role'，可以用来验证
        # assistant_role = choice["message"].get("role", "assistant")

        # 提取 token 使用情况
        llm_output = {}
        if "usage" in response_json:
            llm_output["token_usage"] = response_json["usage"]
            llm_output["model_name"] = response_json.get("model", self.model_name) # API 返回的实际模型名
        
        # 创建 AIMessage 和 ChatGeneration
        ai_message = AIMessage(content=assistant_message_content)
        generation = ChatGeneration(message=ai_message, generation_info=llm_output if llm_output else None)

        return ChatResult(generations=[generation], llm_output=llm_output if llm_output else None)

    # 如果需要异步支持，则需要实现 _agenerate 方法
    # async def _agenerate(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> ChatResult:
    #     # 异步实现...
    #     raise NotImplementedError("异步生成尚未实现。")

