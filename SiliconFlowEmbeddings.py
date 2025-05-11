import requests
from typing import List, Dict, Any
from langchain_core.embeddings import Embeddings
import os # 用于文件读取测试 (如果之前的测试代码还在的话)

# 定义您的硅基流动 API 密钥 (请替换成您的真实密钥)
SILICONFLOW_API_KEY = "sk-cqbudipmjjawthzlkygswcpeaexznqfivulrotjjpvsxlpnb" # <--- 请在这里替换成您的 API 密钥

class SiliconFlowEmbeddings(Embeddings):
    """
    一个与硅基流动 (SiliconFlow) 嵌入服务交互的自定义 LangChain Embeddings 类。
    它使用 netease-youdao/bce-embedding-base_v1 模型，并支持批处理。
    """
    def __init__(self,
                 model_name: str = "netease-youdao/bce-embedding-base_v1",
                 api_key: str = SILICONFLOW_API_KEY,
                 api_url: str = "https://api.siliconflow.cn/v1/embeddings",
                 batch_size: int = 32): # 新增批处理大小参数
        """
        初始化 SiliconFlowEmbeddings 实例。

        参数:
            model_name (str): 要使用的硅基流动模型名称。
            api_key (str): 您的硅基流动 API 密钥。
            api_url (str): 硅基流动嵌入服务的 API 端点。
            batch_size (int): 单次 API 请求中包含的最大文本数量。
        """
        if not api_key:
            raise ValueError("未提供硅基流动 API 密钥。请设置 SILICONFLOW_API_KEY 或在构造函数中传入 api_key。")
        if "sk-" not in api_key and len(api_key) < 50 :
             print(f"警告: API 密钥 '{api_key[:10]}...' 可能格式不正确或不是一个有效的硅基流动API Key。请检查。")

        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.batch_size = batch_size # 存储批处理大小
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _perform_batched_embedding(self, texts_to_embed: List[str]) -> List[List[float]]:
        """
        对有效文本列表进行分批嵌入处理。
        如果任何批次失败，此方法会抛出异常。

        参数:
            texts_to_embed (List[str]): 需要进行嵌入的有效（非空）文本列表。

        返回:
            List[List[float]]: 每个输入文本对应的嵌入向量列表。

        抛出:
            requests.exceptions.RequestException: 如果发生网络或API错误。
            ValueError: 如果API响应格式不正确或数量不匹配。
        """
        all_embeddings_for_valid_texts: List[List[float]] = []
        for i in range(0, len(texts_to_embed), self.batch_size):
            batch = texts_to_embed[i:i + self.batch_size]
            
            payload = {
                "model": self.model_name,
                "input": batch, # API 要求输入不能是空字符串，这里假设 batch 内的文本已验证
                "encoding_format": "float"
            }

            # print(f"DEBUG: 发送批次 {i // self.batch_size + 1}，包含 {len(batch)} 个文本。") # 调试信息

            try:
                response = requests.post(self.api_url, json=payload, headers=self.headers)
                response.raise_for_status()  # 如果 HTTP 请求返回了错误状态码 (4xx or 5xx)，则抛出 HTTPError
                response_json = response.json()

                if "data" not in response_json or not isinstance(response_json["data"], list):
                    raise ValueError(f"API 响应格式错误：未找到 'data' 列表。批次 {i // self.batch_size + 1}。响应: {response_json}")

                embeddings_data = response_json["data"]
                
                if len(embeddings_data) != len(batch):
                    raise ValueError(
                        f"API 返回的嵌入数量 ({len(embeddings_data)}) 与发送的文本数量 ({len(batch)}) 不匹配。"
                        f"批次 {i // self.batch_size + 1}。"
                    )

                current_batch_embeddings = []
                for item_idx, item in enumerate(embeddings_data):
                    if "embedding" not in item or not isinstance(item["embedding"], list):
                        # 如果单个条目没有嵌入或格式不对，也视为批次问题
                        raise ValueError(
                            f"API响应中批次 {i // self.batch_size + 1} 的第 {item_idx +1} 个条目缺少 'embedding' 列表。"
                        )
                    current_batch_embeddings.append(item["embedding"])
                
                all_embeddings_for_valid_texts.extend(current_batch_embeddings)

            except requests.exceptions.RequestException as e:
                print(f"调用硅基流动 API 的批处理在批次 {i // self.batch_size + 1} 失败 (网络错误): {e}")
                raise  # 重新抛出异常，由 embed_documents 统一处理
            except ValueError as e: # 捕获上面我们自己抛出的 ValueError
                print(f"处理嵌入的批处理在批次 {i // self.batch_size + 1} 失败 (数据验证错误): {e}")
                raise # 重新抛出
            except Exception as e: # 捕获其他潜在错误
                print(f"处理嵌入的批处理在批次 {i // self.batch_size + 1} 失败 (未知错误): {e}")
                raise # 重新抛出

        return all_embeddings_for_valid_texts

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文档（文本列表）生成嵌入。
        此方法会处理空字符串输入，并对有效文本进行批处理。

        参数:
            texts (List[str]): 需要生成嵌入的文本列表。

        返回:
            List[List[float]]: 每个文本对应的嵌入向量列表。
                                 如果原始文本为空或嵌入失败，则对应位置为 []。
        """
        if not texts:
            return []

        # 初始化结果列表，所有条目默认为空嵌入
        final_results: List[List[float]] = [[] for _ in texts]

        # 收集所有有效文本及其原始索引
        actual_texts_to_embed: List[str] = []
        original_indices_of_texts_to_embed: List[int] = []

        for idx, text in enumerate(texts):
            if text and text.strip():  # 确保文本非空且非纯空格
                actual_texts_to_embed.append(text)
                original_indices_of_texts_to_embed.append(idx)
        
        # 如果没有有效的文本进行嵌入，直接返回初始化的结果列表
        if not actual_texts_to_embed:
            return final_results
        
        try:
            # 对所有有效文本执行分批嵌入
            embedded_valid_texts = self._perform_batched_embedding(actual_texts_to_embed)
            
            # 将成功获得的嵌入结果放回它们在原始 texts 列表中的位置
            for i, original_idx in enumerate(original_indices_of_texts_to_embed):
                if i < len(embedded_valid_texts): # 确保索引安全
                    final_results[original_idx] = embedded_valid_texts[i]
                else:
                    # 这种情况理论上不应发生，如果 _perform_batched_embedding 保证返回与输入等长的结果
                    print(f"警告: 嵌入结果数量与有效文本数量不匹配。索引 {original_idx} 的嵌入可能丢失。")
                    # 保持 final_results[original_idx] 为 []

            return final_results

        except Exception as e: # 捕获来自 _perform_batched_embedding 的任何异常
            # 如果分批嵌入过程中任何一个环节失败，我们认为整体失败，为所有文本返回空嵌入。
            # _perform_batched_embedding 内部的 print 语句已经打印了具体批次的错误。
            print(f"由于批处理嵌入过程中发生错误 (详情见上述特定批次错误)，将为所有输入文本返回空嵌入。错误摘要: {e}")
            return [[] for _ in texts] # 返回与原始输入等长的空嵌入列表


    def embed_query(self, text: str) -> List[float]:
        """
        为单个查询文本生成嵌入。

        参数:
            text (str): 需要生成嵌入的查询文本。

        返回:
            List[float]: 查询文本对应的嵌入向量，如果文本为空或嵌入失败则返回 []。
        """
        if not text or not text.strip(): # 同样处理空查询
            return []
        
        # embed_documents 方法现在能够处理单个文本的列表（会成为一个批次）
        # 并且会返回一个包含单个结果（或单个空列表）的列表
        result_list = self.embed_documents([text])
        
        # result_list 应该包含一个元素，即查询文本的嵌入或空列表
        return result_list[0] if result_list else []

