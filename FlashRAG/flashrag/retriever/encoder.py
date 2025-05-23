from typing import List
import torch
import numpy as np
from flashrag.retriever.utils import load_model, pooling, parse_query


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, instruction):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)

    @torch.inference_mode(mode=True)
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list,instruction=self.instruction)

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb


class STEncoder:
    def __init__(self, model_name, model_path, max_length, use_fp16, instruction):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model_path = model_path
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.instruction = instruction
        self.model = SentenceTransformer(
            model_path, model_kwargs={"torch_dtype": torch.float16 if use_fp16 else torch.float}
        )

    @torch.inference_mode(mode=True)
    def encode(self, query_list: List[str], batch_size=64, is_query=True) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list,instruction=self.instruction)
        query_emb = self.model.encode(
            query_list, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
        )
        query_emb = query_emb.astype(np.float32, order="C")

        return query_emb

    @torch.inference_mode(mode=True)
    def multi_gpu_encode(self, query_list: List[str], is_query=True, batch_size=None) -> np.ndarray:
        query_list = parse_query(self.model_name, query_list)
        pool = self.model.start_multi_process_pool()
        query_emb = self.model.encode_multi_process(
            query_list, pool, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size
        )
        self.model.stop_multi_process_pool(pool)
        query_emb.astype(np.float32, order="C")

        return query_emb
