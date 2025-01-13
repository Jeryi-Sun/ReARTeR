import os
from typing import List
from copy import deepcopy
import warnings
from tqdm import tqdm
import numpy as np

import asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI
import tiktoken
import time

class OpenaiGenerator:
    """Class for api-based openai models"""

    def __init__(self, config):
        self.model_name = config["generator_model"]
        self.batch_size = config["generator_batch_size"]
        self.generation_params = config["generation_params"]

        self.openai_setting = config["openai_setting"]
        if self.openai_setting["api_key"] is None:
            self.openai_setting["api_key"] = os.getenv("OPENAI_API_KEY")

        if "api_type" in self.openai_setting and self.openai_setting["api_type"] == "azure":
            del self.openai_setting["api_type"]
            self.client = AsyncAzureOpenAI(**self.openai_setting)
        else:
            self.client = AsyncOpenAI(**self.openai_setting)

        #self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    async def get_response(self, input: List, **params):
        for attempt in range(5):
            try:
                response = await self.client.chat.completions.create(model=self.model_name, messages=input, **params)
                n = params.pop("n", 1)
                if n>1:
                    return response.choices
                else:
                    return response.choices[0]
            except Exception as e:
                print(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(5)
                print(e)
                if attempt==4:
                    print("Max attempts reached. Exiting.")
                    raise ValueError("随机失败")
        raise ValueError("随机失败")
            
        
    # async def get_response(self, input: List, **params):
    #     try:
    #         # 处理嵌套的消息格式
    #         def flatten_messages(msg_list):
    #             if isinstance(msg_list, list):
    #                 # 如果是列表，递归处理每个元素
    #                 content = []
    #                 for item in msg_list:
    #                     if isinstance(item, dict):
    #                         if 'content' in item:
    #                             content.append(flatten_messages(item['content']))
    #                     else:
    #                         content.append(str(item))
    #                 return '\n'.join(filter(None, content))
    #             elif isinstance(msg_list, dict):
    #                 # 如果是字典，提取content
    #                 return flatten_messages(msg_list.get('content', ''))
    #             else:
    #                 return str(msg_list)

    #         if isinstance(input, list) and isinstance(input[0], dict):
    #             flattened_content = flatten_messages(input)
    #         else:
    #             flattened_content = ' '.join(map(str, input))

    #         messages = [{
    #             "role": "user",
    #             "content": [{
    #                 "type": "text",
    #                 "text": flattened_content
    #             }]
    #         }]

    #         print("Request messages:", messages)  # 调试信息
            
    #         response = await self.client.chat.completions.create(
    #             model=self.model_name, 
    #             messages=messages,
    #             **params
    #         )
    #         return response.choices[0]
            
    #     except Exception as e:
    #         print(f"API调用错误: {str(e)}")
    #         return f"生成错误: {str(e)}"
    async def get_batch_response(self, input_list: List[List], batch_size, **params):
        total_input = [self.get_response(input, **params) for input in input_list]
        all_result = []
        for idx in range(0, len(input_list), batch_size):
            batch_input = total_input[idx : idx + batch_size]
            batch_result = await asyncio.gather(*batch_input)
            all_result.extend(batch_result)
            print("generated!")
        return all_result

    def generate(self, input_list: List[List], batch_size=None, return_scores=False, **params) -> List[str]:
        # deal with single input
        # if len(input_list) == 1:
        #     input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        # deal with generation params
        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            generation_params.pop("do_sample")

        max_tokens = params.pop("max_tokens", None) or params.pop("max_new_tokens", None)
        stop = params.pop("stop", None)
        generation_params["stop"] = stop
        temperature = params.pop("temperature", 1) or params.pop("temperature", 1)
        generation_params["temperature"] = temperature
        if max_tokens is not None:
            generation_params["max_tokens"] = max_tokens
        else:
            generation_params["max_tokens"] = generation_params.get(
                "max_tokens", generation_params.pop("max_new_tokens", None)
            )
        generation_params.pop("max_new_tokens", None)

        if return_scores:
            if generation_params.get("logprobs") is not None:
                generation_params["logprobs"] = True
                warnings.warn("Set logprobs to True to get generation scores.")
            else:
                generation_params["logprobs"] = True
        if params.get("n") is not None:
            n = params.pop("n")
            generation_params["n"] = n
        else:
            if generation_params.get("n") is not None:
                generation_params["n"] = 1
                warnings.warn("Set n to 1. It can minimize costs.")
            else:
                generation_params["n"] = 1
            n = 1
        

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.get_batch_response(input_list, batch_size, **generation_params))

        # parse result into response text and logprob
        if n==1:
            scores = []
            response_text = []
            for res in result:
                response_text.append(res.message.content)
                if return_scores:
                    score = np.exp(list(map(lambda x: x.logprob, res.logprobs.content)))
                    scores.append(score)
            if return_scores:
                return response_text, scores
            else:
                return response_text
        else:
            scores = []
            response_text = []
            for res in result:
                response_text_list = []
                scores_list = []
                for i in res:
                    response_text_list.append(i.message.content)
                    if return_scores:
                        score = np.exp(list(map(lambda x: x.logprob, i.logprobs.content)))
                        scores_list.append(score)
                scores.append(scores_list)
                response_text.append(response_text_list)
            if return_scores:
                return response_text, scores
            else:
                return response_text

class OpenaiHFEXPGenerator:
    """Class for api-based openai models"""

    def __init__(self, config):
        self.model_name = config["explaner_model_path"]
        self.batch_size = config["generator_batch_size"]
        self.generation_params = config["generation_params"]

        self.openai_setting = config["exp_openai_setting"]
        if self.openai_setting["api_key"] is None:
            self.openai_setting["api_key"] = os.getenv("OPENAI_API_KEY")

        if "api_type" in self.openai_setting and self.openai_setting["api_type"] == "azure":
            del self.openai_setting["api_type"]
            self.client = AsyncAzureOpenAI(**self.openai_setting)
        else:
            self.client = AsyncOpenAI(**self.openai_setting)

        #self.tokenizer = tiktoken.encoding_for_model(self.model_name)

    async def get_response(self, input: List, **params):
        for attempt in range(5):
            try:
                response = await self.client.chat.completions.create(model=self.model_name, messages=input, **params)
                n = params.pop("n", 1)
                if n>1:
                    return response.choices
                else:
                    return response.choices[0]
            except Exception as e:
                print(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(5)
                print(e)
                if attempt==4:
                    print("Max attempts reached. Exiting.")
                    raise ValueError("随机失败")
        raise ValueError("随机失败")
            
        
    # async def get_response(self, input: List, **params):
    #     try:
    #         # 处理嵌套的消息格式
    #         def flatten_messages(msg_list):
    #             if isinstance(msg_list, list):
    #                 # 如果是列表，递归处理每个元素
    #                 content = []
    #                 for item in msg_list:
    #                     if isinstance(item, dict):
    #                         if 'content' in item:
    #                             content.append(flatten_messages(item['content']))
    #                     else:
    #                         content.append(str(item))
    #                 return '\n'.join(filter(None, content))
    #             elif isinstance(msg_list, dict):
    #                 # 如果是字典，提取content
    #                 return flatten_messages(msg_list.get('content', ''))
    #             else:
    #                 return str(msg_list)

    #         if isinstance(input, list) and isinstance(input[0], dict):
    #             flattened_content = flatten_messages(input)
    #         else:
    #             flattened_content = ' '.join(map(str, input))

    #         messages = [{
    #             "role": "user",
    #             "content": [{
    #                 "type": "text",
    #                 "text": flattened_content
    #             }]
    #         }]

    #         print("Request messages:", messages)  # 调试信息
            
    #         response = await self.client.chat.completions.create(
    #             model=self.model_name, 
    #             messages=messages,
    #             **params
    #         )
    #         return response.choices[0]
            
    #     except Exception as e:
    #         print(f"API调用错误: {str(e)}")
    #         return f"生成错误: {str(e)}"
    async def get_batch_response(self, input_list: List[List], batch_size, **params):
        total_input = [self.get_response(input, **params) for input in input_list]
        all_result = []
        for idx in range(0, len(input_list), batch_size):
            batch_input = total_input[idx : idx + batch_size]
            batch_result = await asyncio.gather(*batch_input)
            all_result.extend(batch_result)
            print("generated!")
        return all_result

    def generate(self, input_list: List[List], batch_size=None, return_scores=False, **params) -> List[str]:
        # deal with single input
        # if len(input_list) == 1:
        #     input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size

        # deal with generation params
        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            generation_params.pop("do_sample")

        max_tokens = params.pop("max_tokens", None) or params.pop("max_new_tokens", None)
        stop = params.pop("stop", None)
        generation_params["stop"] = stop
        temperature = params.pop("temperature", 1) or params.pop("temperature", 1)
        generation_params["temperature"] = temperature
        if max_tokens is not None:
            generation_params["max_tokens"] = max_tokens
        else:
            generation_params["max_tokens"] = generation_params.get(
                "max_tokens", generation_params.pop("max_new_tokens", None)
            )
        generation_params.pop("max_new_tokens", None)

        if return_scores:
            if generation_params.get("logprobs") is not None:
                generation_params["logprobs"] = True
                warnings.warn("Set logprobs to True to get generation scores.")
            else:
                generation_params["logprobs"] = True
        if params.get("n") is not None:
            n = params.pop("n")
            generation_params["n"] = n
        else:
            if generation_params.get("n") is not None:
                generation_params["n"] = 1
                warnings.warn("Set n to 1. It can minimize costs.")
            else:
                generation_params["n"] = 1
            n = 1
        

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.get_batch_response(input_list, batch_size, **generation_params))

        # parse result into response text and logprob
        if n==1:
            scores = []
            response_text = []
            for res in result:
                response_text.append(res.message.content)
                if return_scores:
                    score = np.exp(list(map(lambda x: x.logprob, res.logprobs.content)))
                    scores.append(score)
            if return_scores:
                return response_text, scores
            else:
                return response_text
        else:
            scores = []
            response_text = []
            for res in result:
                response_text_list = []
                scores_list = []
                for i in res:
                    response_text_list.append(i.message.content)
                    if return_scores:
                        score = np.exp(list(map(lambda x: x.logprob, i.logprobs.content)))
                        scores_list.append(score)
                scores.append(scores_list)
                response_text.append(response_text_list)
            if return_scores:
                return response_text, scores
            else:
                return response_text
