import re
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from flashrag.utils import get_retriever, get_generator, get_explaner, get_prm, selfask_pred_parse, ircot_pred_parse, rag_reasoning_pred_parse, is_vllm_installed
from flashrag.pipeline import BasicPipeline
from flashrag.dataset import get_batch_dataset, merge_batch_dataset
from flashrag.prompt import PromptTemplate
import random
import json
from flashrag.evaluator.metrics import F1_Score
import torch.nn.functional as F
import time
import torch
class IterativePipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, iter_num=3):
        super().__init__(config, prompt_template)
        self.iter_num = iter_num
        self.generator = get_generator(config)
        self.retriever = get_retriever(config)
        

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        questions = dataset.question
        # run in batch
        past_generation_result = []  # list of N items
        for iter_idx in range(self.iter_num):
            if iter_idx == 0:
                input_query = questions
            else:
                assert len(questions) == len(past_generation_result)
                input_query = [f"{q} {r}" for q, r in zip(questions, past_generation_result)]

            # generation-augmented retrieval
            retrieval_results = self.retriever.batch_search(input_query)
            dataset.update_output(f"retrieval_result_iter_{iter_idx}", retrieval_results)

            # retrieval-augmented generation
            # input_prompts = self.build_prompt(questions, retrieval_results)
            input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(questions, retrieval_results)
            ]

            dataset.update_output(f"prompt_iter_{iter_idx}", input_prompts)
            past_generation_result = self.generator.generate(input_prompts)
            dataset.update_output(f"pred_iter_{iter_idx}", past_generation_result)

        # use last retrieval result for evaluation
        dataset.update_output("retrieval_result", retrieval_results)

        dataset.update_output("pred", past_generation_result)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class SelfRAGPipeline(BasicPipeline):
    # Source: https://github.com/AkariAsai/self-rag
    # The code is released under MIT license

    rel_tokens_names = ["[Irrelevant]", "[Relevant]"]
    retrieval_tokens_names = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
    utility_tokens_names = ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    ground_tokens_names = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
    other_special_tokens = ["<s>", "</s>", "[PAD]", "<unk>", "<paragraph>", "</paragraph>"]
    control_tokens = [
        "[Fully supported]",
        "[Partially supported]",
        "[No support / Contradictory]",
        "[No Retrieval]",
        "[Retrieval]",
        "[Irrelevant]",
        "[Relevant]",
        "<paragraph>",
        "</paragraph>",
        "[Utility:1]",
        "[Utility:2]",
        "[Utility:3]",
        "[Utility:4]",
        "[Utility:5]",
    ]

    task_inst = {
        "wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
        "fever": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
        "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
        "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
        "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
        "arc_c": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
        "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
        "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.",
        "normal_qa": "Answer the following question, give me a short answer.",
    }

    def __init__(
        self,
        config,
        threhsold=0.2,
        max_depth=2,
        beam_width=2,
        w_rel=1.0,
        w_sup=1.0,
        w_use=1.0,
        use_grounding=True,
        use_utility=True,
        use_seqscore=True,
        ignore_cont=True,
        mode="adaptive_retrieval",
        prompt_template=None,
    ):

        super().__init__(config, prompt_template)
        self.generator = get_generator(config)
        self.retriever = get_retriever(config)

        assert mode in ["adaptive_retrieval", "always_retrieve", "no_retrieval"]

        self.task = config["dataset_name"]
        self.task_instruction = self.task_inst.get(self.task, self.task_inst["normal_qa"])
        if self.task_instruction is not None:
            question_inst = self.task_instruction + "\n\n## Input:\n\n{question}"
        else:
            question_inst = "{question}"
        if prompt_template is None:
            self.prompt_template = PromptTemplate(
                config, user_prompt="### Instruction:\n" + question_inst + "\n\n### Response:\n", enable_chat=False
            )

        self.threshold = threhsold
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.w_rel, self.w_sup, self.w_use = w_rel, w_sup, w_use
        self.use_grounding = use_grounding
        self.use_utility = use_utility
        self.use_seqscore = use_seqscore
        self.ignore_cont = ignore_cont
        self.mode = mode
        self.closed = self.task in ["fever", "arc_c"]
        tokenizer = AutoTokenizer.from_pretrained(config["generator_model_path"], padding_side="left")
        self.ret_tokens, self.rel_tokens, self.grd_tokens, self.ut_tokens = self.load_special_tokens(
            tokenizer, use_grounding=use_grounding, use_utility=use_utility
        )

    def load_special_tokens(self, tokenizer, use_grounding, use_utility):
        ret_tokens = {token: tokenizer.convert_tokens_to_ids(token) for token in self.retrieval_tokens_names}
        rel_tokens = {}
        for token in ["[Irrelevant]", "[Relevant]"]:
            rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        grd_tokens = None
        if use_grounding is True:
            grd_tokens = {}
            for token in self.ground_tokens_names:
                grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        ut_tokens = None
        if use_utility is True:
            ut_tokens = {}
            for token in self.utility_tokens_names:
                ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

        return ret_tokens, rel_tokens, grd_tokens, ut_tokens

    def judge_retrieve(self, input_prompts):
        """Calculate whether a retrieve is required based on the output probability of
        the special token in the model"""

        if self.mode != "always_retrieve":
            # result for total batch
            all_pred_token_ids = []
            all_pred_text = []
            all_pred_log_probs = []
            preds = self.generator.generate(input_prompts, return_raw_output=True, logprobs=32000)
            for single_pred in preds:
                pred_token_ids = single_pred.outputs[0].token_ids
                pred_text = single_pred.outputs[0].text
                pred_log_probs = single_pred.outputs[0].logprobs
                all_pred_token_ids.append(pred_token_ids)
                all_pred_text.append(pred_text)
                all_pred_log_probs.append(pred_log_probs)

        if self.mode == "always_retrieve":
            retrieval_flags = [True] * len(input_prompts)

        elif self.mode == "no_retrieval":
            retrieval_flags = [False] * len(input_prompts)

        else:
            retrieval_flags = []
            for idx, single_pred in enumerate(preds):
                if self.threshold is not None:
                    score_dict = {}
                    for tok, tok_id in self.ret_tokens.items():
                        if tok_id not in all_pred_log_probs[idx][0]:
                            score_dict[tok] = -100
                        prob = all_pred_log_probs[idx][0][tok_id].logprob
                        score_dict[tok] = float(prob)
                    do_retrieve = (
                        score_dict["[Retrieval]"] / (score_dict["[Retrieval]"] + score_dict["[No Retrieval]"])
                        > self.threshold
                    )
                else:
                    do_retrieve = "[Retrieval]" in all_pred_text[idx]

                retrieval_flags.append(do_retrieve)

        return retrieval_flags

    def critic_preds(self, preds):
        """Evaluate predictions using different retrieval docs"""

        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        results = {}
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / max(len(pred.outputs[0].token_ids), 1)
            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # Compute reward scores
            for tok, id in self.rel_tokens.items():
                prob = pred_log_probs[0][id].logprob if id in pred_log_probs[0] else -100
                relevance_score_dict[p_idx][tok] = np.exp(float(prob))

            if self.grd_tokens is not None:
                groundness_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(self.grd_tokens.values()):
                        groundness_token_appear_indices.append(tok_idx)
                        break
                if len(groundness_token_appear_indices) > 0:
                    idx = groundness_token_appear_indices[0]
                    for token, token_id in self.grd_tokens.items():
                        prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100
                        grd_score_dict[p_idx][token] = np.exp(float(prob))
            utility_token_appear_indices = []
            if self.ut_tokens is not None:
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(self.ut_tokens.values()):
                        utility_token_appear_indices.append(tok_idx)
                if len(utility_token_appear_indices) > 0:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in self.ut_tokens.items():
                        prob = pred_log_probs[idx][token_id].logprob if token_id in pred_log_probs[idx] else -100
                        ut_score_dict[p_idx][token] = np.exp(float(prob))

            relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                np.sum(list(relevance_score_dict[p_idx].values()))
            )

            if len(grd_score_dict[p_idx]) == 3:
                gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
                    grd_score_dict[p_idx]["[Partially supported]"] / gt_sum
                )
            else:
                ground_score = 0.0

            if len(ut_score_dict[p_idx]) == 5:
                ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum(
                    [
                        ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i + 1)] / ut_sum)
                        for i in range(len(ut_scores))
                    ]
                )
            else:
                utility_score = 0.0

            if self.use_seqscore is True:
                final_score = (
                    np.exp(seq_score)
                    + self.w_rel * relevance_score
                    + self.w_sup * ground_score
                    + self.w_use * utility_score
                )
            else:
                final_score = self.w_rel * relevance_score + self.w_sup * ground_score + self.w_use * utility_score

            overall_scores[p_idx] = {
                "final_score": final_score,
                "relevance_score": relevance_score,
                "ground_score": ground_score,
                "utility_score": utility_score,
                "relevance_score_dict": relevance_score_dict,
                "grd_score_dict": grd_score_dict,
                "ut_score_dict": utility_score,
            }
            results["retrieval_{}".format(p_idx)] = {"pred": pred_text, "score": final_score}

        # modify and add do retrieve tokens (only used in long-form generation)
        final_preds = []
        if "[No Retrieval]" in pred_text:
            ret_token_appear_indices = []
            substrings = pred_text.split("[No Retrieval]")

            for tok_idx, tok in enumerate(pred_token_ids):
                if tok == self.ret_tokens["[No Retrieval]"]:
                    ret_token_appear_indices.append(tok_idx)

            ret_token_score_dict = {}
            retrieval_remap = {}
            for order, idx in enumerate(ret_token_appear_indices):
                ret_token_score_dict.setdefault(order, {})
                for tok, tok_id in self.ret_tokens.items():
                    prob = pred_log_probs[idx][tok_id].logprob if tok_id in pred_log_probs[idx] else -100
                    ret_token_score_dict[order][tok] = np.exp(prob)
                if ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"] != 0.0:
                    do_retrieve = (
                        ret_token_score_dict[order]["[Retrieval]"]
                        + ret_token_score_dict[order]["[Continue to Use Evidence]"]
                    ) / (
                        ret_token_score_dict[order]["[Retrieval]"] + ret_token_score_dict[order]["[No Retrieval]"]
                    ) > self.threshold
                else:
                    do_retrieve = 0.0
                if do_retrieve > self.threshold:
                    retrieval_remap[order] = True
                else:
                    retrieval_remap[order] = False
            processed_pred = ""
            for substr_i, substring in enumerate(substrings):
                if substr_i in retrieval_remap and retrieval_remap[substr_i] is True:
                    processed_pred += substring + "[Retrieval]"
                else:
                    processed_pred += substring + "[No Retrieval]"
            pred_text = processed_pred
            final_preds.append(pred_text)
        else:
            final_preds.append(pred_text)

        scores = [overall_scores[p_idx]["final_score"] for p_idx in overall_scores]

        return results, final_preds, scores, overall_scores

    def postprocess_prediction(self, pred):
        def fix_spacing(input_text):
            # Add a space after periods that lack whitespace
            output_text = re.sub(r"(?<=\w)([.!?])(?=\w)", r"\1 ", input_text)
            return output_text

        for token in self.control_tokens:
            pred = pred.replace(token, "")
        if "</s>" in pred:
            pred = pred.replace("</s>", "")
        if "\n" in pred:
            pred = pred.replace("\n", "")
        if "<|endoftext|>" in pred:
            pred = pred.replace("<|endoftext|>", "")

        pred = pred.strip()
        if type(pred) is str and pred[0] == "#" or pred[0] == ":":
            pred = pred[1:]
        if len(pred) == 0:

            return ""

        return fix_spacing(pred)

    def select_best_prediction(self, results):
        answer2score = {}
        if self.closed is True:
            for key, result in results.items():
                answer = self.postprocess_prediction(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                answer2score[answer] += score
            sorted_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)
            best_pred = sorted_answers[0][0]
        else:
            path2score = {key: item["score"] for key, item in results.items() if key != "no_retrieval"}
            best_path = sorted(path2score.items(), key=lambda x: x[1], reverse=True)[0][0]
            best_pred = results[best_path]["pred"]

        return best_pred

    def run_single_beam(self, prompt, item_retrieval_result=None):
        curr_depth = 1
        terminated = False
        node_id = 0
        prediction_tree = {}
        levels = {}
        prediction_tree[node_id] = {
            "prompt": prompt,
            "pred": "[Retrieval]",
            "processed_pred": "",
            "score": None,
            "ctx": None,
            "parent": None,
        }
        levels[0] = [0]
        while curr_depth < self.max_depth:
            levels[curr_depth] = []
            if curr_depth - 1 in levels and terminated is False:
                for node in levels[curr_depth - 1]:
                    pred = prediction_tree[node]["pred"]
                    if pred == "</s>":
                        terminated = True
                        continue
                    prompt = prediction_tree[node]["prompt"]
                    prev_generation = prediction_tree[node]["processed_pred"]
                    score = prediction_tree[node]["score"]
                    if "[Retrieval]" in pred:
                        retrieval_results = {}

                        if item_retrieval_result is not None:
                            aug_prompts = [
                                prompt
                                + prev_generation
                                + "[Retrieval]"
                                + "<paragraph>{}</paragraph>".format(para["contents"])
                                for para in item_retrieval_result
                            ]
                        else:
                            aug_prompts = [prompt + prev_generation]

                        item_pred = self.generator.generate(aug_prompts, return_raw_output=True)
                        _, preds, scores, overall_score_dict = self.critic_preds(item_pred)

                        for i, (pred, p_score) in enumerate(zip(preds, scores)):
                            retrieval_results[i] = {"pred": pred, "score": p_score}

                        for i, result in retrieval_results.items():
                            node_id += 1
                            node_score = result["score"] * score if score is not None else result["score"]
                            pred = result["pred"]
                            prediction_tree[node_id] = {
                                "prompt": prompt + prev_generation,
                                "pred": pred,
                                "score": node_score,
                                "ctx": item_retrieval_result[i],
                                "parent": node,
                                "overall_score_dict": overall_score_dict,
                            }

                            if "[Retrieval]" in pred:
                                gen_result_index = pred.index("[Retrieval]")
                                prev_generation = pred[:gen_result_index]
                            else:
                                prev_generation = pred
                            prediction_tree[node_id]["processed_pred"] = prev_generation
                            levels[curr_depth].append(node_id)

                current_rank = levels[curr_depth]
                node2score = {node_id: prediction_tree[node_id]["score"] for node_id in current_rank}
                top_nodes = sorted(node2score.items(), key=lambda x: x[1], reverse=True)[: self.beam_width]
                levels[curr_depth] = [node[0] for node in top_nodes]
                curr_depth += 1
            else:
                break

        final_prediction = ""
        parent = 0
        best_selections = {}

        # Traverse from the bottom
        levels = {k: v for k, v in levels.items() if len(v) > 0 and k != 0}
        for path_i, node in enumerate(levels[len(levels)]):
            if node == 0:
                break
            best_selections[path_i] = [node]
            current_node = node
            current_level = curr_depth
            if current_node is None:
                continue
            while current_level > 0 and current_node is not None:
                parent = prediction_tree[current_node]["parent"]
                best_selections[path_i] = [parent] + best_selections[path_i]
                current_node = parent
                current_level += 1

        final_prediction = {}
        splitted_sentences = {}
        original_splitted_sentences = {}
        ctxs = {}
        for path_i, nodes in best_selections.items():
            final_prediction[path_i] = " ".join(
                [
                    prediction_tree[node]["processed_pred"]
                    for node in nodes
                    if node is not None
                    and (
                        self.ignore_cont is False
                        or (
                            self.ignore_cont is True
                            and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
                        )
                    )
                ]
            )
            splitted_sentences[path_i] = [
                prediction_tree[node]["processed_pred"]
                for node in nodes
                if node is not None
                and (
                    self.ignore_cont is False
                    or (
                        self.ignore_cont is True
                        and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
                    )
                )
            ]
            original_splitted_sentences[path_i] = [
                prediction_tree[node]["pred"]
                for node in nodes
                if node is not None
                and (
                    self.ignore_cont is False
                    or (
                        self.ignore_cont is True
                        and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
                    )
                )
            ]
            ctxs[path_i] = [
                prediction_tree[node]["ctx"]
                for node in nodes
                if node is not None
                and (
                    self.ignore_cont is False
                    or (
                        self.ignore_cont is True
                        and "[No support / Contradictory]" not in prediction_tree[node]["processed_pred"]
                    )
                )
            ]

        result = {
            "final_prediction": final_prediction,
            "splitted_sentences": splitted_sentences,
            "original_splitted_sentences": original_splitted_sentences,
            "best_selections": best_selections,
            "ctxs": ctxs,
            "prediction_tree": prediction_tree,
        }

        return final_prediction[0], result

    def postprocess_long_form(self, pred, intermediate):
        final_output = ""
        docs = []
        prev_gen = []
        if "splitted_sentences" not in intermediate:
            final_output = self.postprocess_prediction(pred)
        else:
            if len(self.postprocess_prediction(pred)) == 0:
                intermediate["splitted_sentences"][0], intermediate["ctxs"][0] = (
                    intermediate["splitted_sentences"][1],
                    intermediate["ctxs"][1],
                )
            for idx, (sent, doc) in enumerate(zip(intermediate["splitted_sentences"][0], intermediate["ctxs"][0])):
                if len(sent) == 0:
                    continue
                postprocessed_result = self.postprocess_prediction(sent)
                if postprocessed_result in prev_gen:
                    continue
                else:
                    prev_gen.append(postprocessed_result)
                final_output += postprocessed_result[:-1] + " [{}]".format(idx) + ". "
                docs.append(doc)
            if len(final_output) == 0:
                final_output = final_output
            if len(final_output) > 0 and final_output[-1] == " ":
                final_output = final_output[:-1]
            final_output = final_output.strip()
            final_output = final_output.replace(".[Continue to Use Evidence]", " [1]. ")
            final_output = final_output.replace(". [1] ", " [1]. ")

        return final_output

    def run_batch_pred_long_form(self, dataset):
        questions = dataset.question
        retrieval_results = self.retriever.batch_search(questions)
        dataset.update_output("retrieval_result", retrieval_results)

        # input_prompts = self.build_prompt(questions)
        input_prompts = [self.prompt_template.get_string(question=q) for q in questions]

        # determine whether to retrieve
        retrieval_flags = self.judge_retrieve(input_prompts)
        dataset.update_output("retrieval_flag", retrieval_flags)

        # for long form task, only support single item run
        for item, prompt, retrieval_flag in zip(dataset, input_prompts, retrieval_flags):
            if retrieval_flag:
                pred, intermediate_result = self.run_single_beam(prompt, item_retrieval_result=item.retrieval_result)
                item.update_output("intermediate_result", intermediate_result)

                if self.task == "factscore":
                    pred = self.postprocess_prediction(pred)
                else:
                    assert self.task in ["asqa", "eli5"]
                    pred = self.postprocess_long_form(pred, intermediate_result)
            else:
                prompt += "[No Retrieval]"
                pred = self.generator.generate(prompt)[0]

            item.update_output("pred", pred)

        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None, batch_size=256, long_form=False):
        all_dataset_list = []
        run_func = self.run_batch_pred_long_form if long_form else self.run_batch_pred
        # to avoid oom
        for batch_dataset in tqdm(get_batch_dataset(dataset, batch_size=batch_size), desc="Batch dataset: "):
            batch_dataset = run_func(batch_dataset)
            all_dataset_list.append(batch_dataset)
        dataset = merge_batch_dataset(all_dataset_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run_batch_pred(self, dataset):
        questions = dataset.question
        retrieval_results = self.retriever.batch_search(questions)
        dataset.update_output("retrieval_result", retrieval_results)

        # input_prompts = self.build_prompt(questions)
        input_prompts = [self.prompt_template.get_string(question=q) for q in questions]

        # determine whether to retrieve
        retrieval_flags = self.judge_retrieve(input_prompts)
        dataset.update_output("retrieval_flag", retrieval_flags)

        # process input item based on whether to retrieve
        all_input_list = []
        for idx, (prompt, item) in enumerate(zip(input_prompts, dataset)):
            retrieval_flag = retrieval_flags[idx]

            if retrieval_flag:
                retrieval_result = retrieval_results[idx]
                # for each doc in retrieval result, there is a prompt as input
                prompt_list = [
                    prompt + "[Retrieval]<paragraph>{}</paragraph>".format(para["contents"])
                    for para in retrieval_result
                ]
            else:
                prompt += "[No Retrieval]"
                prompt_list = [prompt]

            item.update_output("prompt", prompt_list)
            all_input_list += prompt_list

        batch_pred = self.generator.generate(all_input_list, return_raw_output=True, logprobs=32016)

        # parse output based on retrieval flag
        pred_idx = 0
        pred_answer_list = []
        for idx, (retrieval_flag, item) in enumerate(zip(retrieval_flags, dataset)):
            if retrieval_flag:
                # for item that need retrieval, there may have more than one prediction
                item_pred = batch_pred[pred_idx : pred_idx + len(retrieval_results[idx])]
                pred_idx += len(retrieval_results[idx])
                critic_result, _, _, _ = self.critic_preds(item_pred)
                item.update_output("critic_result", critic_result)

                # select best prediction
                pred = self.select_best_prediction(critic_result)

            else:
                item_pred = batch_pred[pred_idx : pred_idx + 1][0]
                pred_idx += 1
                pred = item_pred.outputs[0].text

            pred = self.postprocess_prediction(pred)
            pred_answer_list.append(pred)

        dataset.update_output("pred", pred_answer_list)

        return dataset


class FLAREPipeline(BasicPipeline):
    def __init__(
        self,
        config,
        threshold=0.2,
        look_ahead_steps=64,
        max_generation_length=256,
        max_iter_num=5,
        prompt_template=None,
    ):
        super().__init__(config, prompt_template)

        self.generator = get_generator(config)
        self.retriever = get_retriever(config)

        self.threshold = threshold
        self.max_generation_length = max_generation_length
        self.max_iter_num = max_iter_num
        self.look_ahead_steps = look_ahead_steps
        self.stop_sym = list("!@#$%^&*()\n\n)(*&^%$#@!")

    def get_next_sentence(self, output, scores):
        tokenizer = self.generator.tokenizer
        text_sentences = re.split(r"(?<=[^A-Z].[.?]) +", output)
        if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            token_id_sentences = [tokenizer.encode(s, add_special_tokens=False) for s in text_sentences]
        else:
            token_id_sentences = [tokenizer.encode(s, allowed_special="all") for s in text_sentences]

        output_ids = tokenizer.encode(output, add_special_tokens=False)

        # assert sum([len(s) for s in token_id_sentences]) == len(
        #    output_ids), "token id sentences length not equal to output ids length"

        first_sent_ids = token_id_sentences[0]
        first_sent_score = scores[: len(first_sent_ids)]

        return text_sentences[0], first_sent_score

    def judge_sent_confidence(self, sent, sent_score):
        judge_result = all([score > self.threshold for score in sent_score])
        new_query = None
        if not judge_result:
            tokenizer = self.generator.tokenizer
            if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                sent_ids = tokenizer.encode(sent, add_special_tokens=False)
            else:
                sent_ids = tokenizer.encode(sent, allowed_special="all")
            # assert len(sent_ids) == len(sent_score)
            new_query_ids = [i for i, score in zip(sent_ids, sent_score) if score > self.threshold]
            new_query = tokenizer.decode(new_query_ids)
            if len(new_query) == 0:
                judge_result = True
        return judge_result, new_query

    def run_item(self, item):
        question = item.question
        gen_length = 0
        iter_round = 0
        final_gen_result = ""
        while gen_length < self.max_generation_length and iter_round < self.max_iter_num:
            input_prompt = self.prompt_template.get_string(question=question, previous_gen=final_gen_result)

            # input_prompt = self.build_prompt(
            #     question_list=[question], use_reference=False, previous_gen=final_gen_result)[0]
            # scores: token logits of the whole generation seq
            round_gen_output, scores = self.generator.generate(
                input_prompt, return_scores=True, stop=self.stop_sym, max_new_tokens=self.look_ahead_steps
            )
            round_gen_output, scores = round_gen_output[0], scores[0]
            # next_sent_scores: token logits of the first sent in generation seq
            next_sent, next_sent_score = self.get_next_sentence(round_gen_output, scores)
            # judge next sentence
            judge_result, query = self.judge_sent_confidence(next_sent, next_sent_score)
            item.update_output(f"judge_result_iter{iter_round}", judge_result)

            if not judge_result:
                # do retrieval-augmented generation
                retrieval_result = self.retriever.search(query)
                item.update_output("retrieval_result", retrieval_result)
                input_prompt = self.prompt_template.get_string(
                    question=question, retrieval_result=retrieval_result, previous_gen=final_gen_result
                )

                # input_prompt = self.build_prompt(
                #     question_list = [question],
                #     retrieval_results = [retrieval_result],
                #     previous_gen = final_gen_result)[0]
                output, scores = self.generator.generate(
                    input_prompt, return_scores=True, stop=self.stop_sym, max_new_tokens=self.look_ahead_steps
                )
                output, scores = output[0], scores[0]
                next_sent, _ = self.get_next_sentence(output, scores)
                item.update_output(f"gen_iter_{iter_round}", next_sent)
                item.update_output("retrieval_result", retrieval_result)

            final_gen_result += next_sent
            gen_length += len(next_sent_score)
            iter_round += 1

        item.update_output("pred", final_gen_result)

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        for item in tqdm(dataset, desc="Inference: "):
            self.run_item(item)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset


class SelfAskPipeline(BasicPipeline):
    FOLLOW_UP_PATTERN = r"Follow up:.*\n"

    def __init__(self, config, prompt_template=None, max_iter=5, single_hop=True):
        super().__init__(config, prompt_template)
        from flashrag.prompt.selfask_examplars import SELF_ASK_PROMPT_SINGLE_HOP, SELF_ASK_PROMPT_MULTI_HOP

        self.generator = get_generator(config)
        self.retriever = get_retriever(config)

        self.single_hop = single_hop
        self.max_iter = max_iter
        self.P_INS = SELF_ASK_PROMPT_SINGLE_HOP if self.single_hop else SELF_ASK_PROMPT_MULTI_HOP

    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Context{idx+1}: {text}\n"

        return format_reference

    def _remove_duplicate_doc(self, docs):
        assert all(["id" in doc for doc in docs])
        new_doc_list = []
        exist_ids = []
        for doc in docs:
            doc_id = doc["id"]
            if doc_id not in exist_ids:
                exist_ids.append(doc_id)
                new_doc_list.append(doc)
        return new_doc_list

    def run_item(self, item):
        question = item.question
        retrieval_result = self.retriever.search(question)

        stop_condition = "Intermediate answer:"
        follow_ups = "No." if self.single_hop else "Yes."
        res = ""
        early_exit = False
        for idx in range(self.max_iter):
            input_prompt = (
                self.P_INS
                + "\n"
                + self.format_reference(retrieval_result)
                + f"\nQuesiton: {question}"
                + "\nAre follow up questions needed here: "
                + follow_ups
                + "\n"
                + res
            )
            gen_out = self.generator.generate(input_prompt, stop=["Context:", "#", stop_condition])[0]
            item.update_output(f"intermediate_output_iter{idx}", gen_out)

            if stop_condition == "Intermediate answer:":
                res += gen_out.split("Intermediate answer:")[0]
                stop_condition = "Follow up:"

            elif stop_condition == "Follow up:":
                followup_split = re.split(self.FOLLOW_UP_PATTERN, gen_out)
                res += followup_split[0]

                if len(followup_split) > 1:
                    res += re.findall(self.FOLLOW_UP_PATTERN, gen_out)[0]
                stop_condition = "Intermediate answer:"

            # make sure the result does not end in a new line
            if len(res) == 0:
                early_exit = True
                break
            if res[-1] == "\n":
                res = res[:-1]

            if "Follow up: " in gen_out:
                # get the first follow up
                new_query = [l for l in gen_out.split("\n") if "Follow up: " in l][0].split("Follow up: ")[-1]
                retrieval_result = self.retriever.search(new_query)

            if "So the final answer is: " in gen_out:
                res = (
                    self.format_reference(retrieval_result)
                    + f"\nQuesiton: {question}"
                    + "\nAre follow up questions needed here: "
                    + follow_ups
                    + "\n"
                    + res
                )
                early_exit = True
                # print("Success: early exit!")
                break

        if not early_exit:
            res = (
                self.format_reference(retrieval_result)
                + f"\nQuesiton: {question}"
                + "\nAre follow up questions needed here: "
                + follow_ups
                + "\n"
                + res
            )

        item.update_output("retrieval_result", retrieval_result)
        item.update_output("pred", res)

    def run(self, dataset, do_eval=True, pred_process_fun=selfask_pred_parse):
        for item in tqdm(dataset, desc="Inference: "):
            self.run_item(item)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset


class IRCOTPipeline(BasicPipeline):
    IRCOT_INSTRUCTION = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'
    IRCOT_EXAMPLE = "Wikipedia Title: Kurram Garhi\nKurram Garhi is a small village located near the city of Bannu, which is the part of Khyber Pakhtunkhwa province of Pakistan. Its population is approximately 35000. Barren hills are near this village. This village is on the border of Kurram Agency. Other nearby villages are Peppal, Surwangi and Amandi Kala.\n\nWikipedia Title: 2001–02 UEFA Champions League second group stage\nEight winners and eight runners- up from the first group stage were drawn into four groups of four teams, each containing two group winners and two runners- up. Teams from the same country or from the same first round group could not be drawn together. The top two teams in each group advanced to the quarter- finals.\n\nWikipedia Title: Satellite tournament\nA satellite tournament is either a minor tournament or event on a competitive sporting tour or one of a group of such tournaments that form a series played in the same country or region.\n\nWikipedia Title: Trojkrsti\nTrojkrsti is a village in Municipality of Prilep, Republic of Macedonia.\n\nWikipedia Title: Telephone numbers in Ascension Island\nCountry Code:+ 247< br> International Call Prefix: 00 Ascension Island does not share the same country code( +290) with the rest of St Helena.\n\nQuestion: Are both Kurram Garhi and Trojkrsti located in the same country?\nThought: Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country. So the answer is: no.\n\n"

    def __init__(self, config, prompt_template=None, retriever=None, generator=None, max_iter=2):
        # if not provide prompt template, use default template provided by IRCOT
        if prompt_template is None:
            prompt_template = PromptTemplate(
                config=config,
                system_prompt=f"{self.IRCOT_INSTRUCTION}\n\n{self.IRCOT_EXAMPLE}",
                user_prompt="{reference}Question: {question}\nThought:",
                reference_template="Wikipedia Title: {title}\n{text}\n\n",
                enable_chat=False,
            )

        super().__init__(config, prompt_template)
        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever

        self.max_iter = max_iter

    def run_item(self, item):
        question = item.question
        retrieval_result, scores = self.retriever.search(question, return_score=True)
        doc2score = {doc_item["id"]: score for doc_item, score in zip(retrieval_result, scores)}
        id2doc = {doc_item["id"]: doc_item for doc_item in retrieval_result}

        thoughts = []
        iter_num = 0
        while iter_num < self.max_iter:
            input_prompt = self.prompt_template.get_string(
                question=question, retrieval_result=retrieval_result, previous_gen=" ".join(thoughts)
            )
            new_thought = self.generator.generate(input_prompt)[0]
            thoughts.append(new_thought)
            iter_num += 1
            if "So the answer is:" in new_thought:
                item.update_output(
                    f"intermediate_output_iter{iter_num}",
                    {
                        "input_prompt": input_prompt,
                        "new_thought": new_thought,
                    },
                )
                break

            # retrieve new docs and merge
            new_retrieval_result, new_scores = self.retriever.search(new_thought, return_score=True)
            for doc_item, score in zip(new_retrieval_result, new_scores):
                id2doc[doc_item["id"]] = doc_item
                doc_id = doc_item["id"]
                if doc_id in doc2score:
                    doc2score[doc_id] = max(doc2score[doc_id], score)
                else:
                    doc2score[doc_id] = score
            sorted_doc_score = sorted(doc2score.items(), key=lambda x: x[1], reverse=False)
            sorted_doc_id = [t[0] for t in sorted_doc_score]
            retrieval_result = [id2doc[id] for id in sorted_doc_id]

            item.update_output(
                f"intermediate_output_iter{iter_num}",
                {
                    "input_prompt": input_prompt,
                    "new_thought": new_thought,
                    "new_retreival_result": new_retrieval_result,
                },
            )

        item.update_output("retrieval_result", retrieval_result)
        item.update_output("pred", " ".join(thoughts))
        return item

    def run(self, dataset, do_eval=True, pred_process_fun=ircot_pred_parse):
        for item in tqdm(dataset, desc="Inference: "):
            self.run_item(item)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset


class ReasoningPipeline(BasicPipeline):
    FOLLOW_UP_PATTERN = r"Follow up:.*\n"
    INTER_ANS = r"Intermediate answer:.*\n"

    def __init__(self, config, prompt_template=None, max_iter=5):
        super().__init__(config, prompt_template)
        from flashrag.prompt.reasoning_examplars import REASON_PROMPT, EXP_PROMPT, CORRECT_PROMPT, EXP_SYS_PROMPT, SUMMATY_PROMPT

        self.generator = get_generator(config)
        self.retriever = get_retriever(config)
        if hasattr(config, 'explaner_model_path'):
            self.explaner = get_explaner(config)
        if hasattr(config, 'prm_path'):
            self.prm = get_prm(config)

        self.max_iter = max_iter # 这个用法是，不超过 5 个Follow up question 出现就继续否则就停止
        
        self.P_INS = REASON_PROMPT
        self.EXP = EXP_PROMPT
        self.COR = CORRECT_PROMPT
        self.EXP_SYS = EXP_SYS_PROMPT
        self.SUM = SUMMATY_PROMPT
        

        self.max_tokens = 200
        self.temperature_range = [0.7, 1.0]

        self.metric = F1_Score(config)

    def format_reference(self, retrieval_result):
        format_reference = "\nRetrieved documents:\n"
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc{idx+1}: {text}\n"

        return format_reference
    def run_item_mcts(self, question, partial_answer, beta=0.0):
        question = question + '\n' + partial_answer
        all_retrieval_result = []
        iter_count = 0
        accumulated_output = ""
        bad_gen = False
        gen_num = 0

        while iter_count < self.max_iter:
            # Build the prompt with the question and any accumulated output            
            system_prompt = self.P_INS
            
            user_prompt = (
                f"#\nQuestion: {question}\n"
                + accumulated_output
            )
            messages = [
                [{
                "role": "system",
                "content": system_prompt
                },
                {
                "role": "user",
                "content":  user_prompt
                }]]

            # Generate the output with stop conditions
            temperature = random.choice(self.temperature_range)
            gen_out = self.generator.generate(messages, stop=["Doc", "#"], temperature=temperature)[0]
            gen_num += 1
            # Check if the generation indicates that retrieval is needed for the last follow-up question
            stop_condition = "Search: Yes."
            if stop_condition in gen_out:
                gen_out = gen_out.split(stop_condition)[0] + stop_condition
                # Extract the last follow-up question from the generated output
                follow_up_questions = re.findall(self.FOLLOW_UP_PATTERN, gen_out)
                if follow_up_questions:
                    accumulated_output += gen_out  # Accumulate output for continuity in next iteration
                    latest_follow_up = follow_up_questions[-1].replace("Follow up: ", "").strip()
                    # Perform retrieval based on the latest follow-up question
                    retrieval_result = self.retriever.search(latest_follow_up)
                    all_retrieval_result.append(retrieval_result)
                    # Format and add retrieval results to accumulated_output
                    formatted_reference = self.format_reference(retrieval_result)
                    accumulated_output += formatted_reference+"\nIntermediate answer:"
                else: 
                    accumulated_output_copy = accumulated_output
                    parts = accumulated_output.rsplit("Search: ", 1)
                    if accumulated_output == "": # 模型需要直接回答
                        retrieval_result = self.retriever.search(question)
                        all_retrieval_result.append(retrieval_result)
                        # Format and add retrieval results to accumulated_output
                        formatted_reference = self.format_reference(retrieval_result)
                        accumulated_output += formatted_reference
                        accumulated_output += "\nSo the final answer is:"
                    else:
                        modified_last_part = "No."+"\nRetrieved documents:" + "\nIntermediate answer:"
                        accumulated_output = parts[0] + "Search: " + modified_last_part
                        if accumulated_output_copy == accumulated_output: # 陷入死循环了，采用直接回答终止
                            retrieval_result = self.retriever.search(question)
                            all_retrieval_result.append(retrieval_result)
                            # Format and add retrieval results to accumulated_output
                            formatted_reference = self.format_reference(retrieval_result)
                            accumulated_output = ""
                            accumulated_output += formatted_reference
                            accumulated_output += "\nSo the final answer is:"
                    print(accumulated_output)
                
                    
                # Update iter_count based on the number of follow-ups in the generated answer
                iter_count += len(follow_up_questions)
            elif "So the final answer is:" in gen_out or "So the final answer is:" in accumulated_output:
                # Final answer is generated, store and exit loop
                accumulated_output += gen_out  # Accumulate output for continuity in next iteration
                break
            else:
                accumulated_output += gen_out
                if iter_count == self.max_iter or gen_num==2*self.max_iter:
                    accumulated_output += "So the final answer is:"
                else:
                    accumulated_output += "Follow up:"
            
            # Check if iteration count exceeds the max iteration limit
            if iter_count > self.max_iter or gen_num>2*self.max_iter:
                follow_up_questions = re.findall(self.INTER_ANS, accumulated_output)
                final_output = ""
                for inter_ans in follow_up_questions:
                    final_output += inter_ans.replace("Intermediate answer:", "").strip()
                break

        # Store retrieval results for reference
        reasoning_step = self.extract_reasoning_steps(accumulated_output)
        if reasoning_step == []:
            bad_gen = True
        if hasattr(self, 'prm_score') and hasattr(self, 'critic_exp_model'):
            for idx, _ in enumerate(reasoning_step):
                # prm
                score = self.prm_score(question,"\n".join(reasoning_step[:idx]), reasoning_step[idx])
                # 纠错
                if score<beta:
                    explanation = self.critic_exp_model(question, "\n".join(reasoning_step[:idx]), reasoning_step[idx], "\n".join(reasoning_step[idx+1:]) if len(reasoning_step)>idx else "", score)
                    
                    new_reasoning_step, new_score, _ = self.correction_step(question, "\n".join(reasoning_step[:idx]), reasoning_step[idx], explanation, score)
                    if new_score>score:
                        reasoning_step[idx] = new_reasoning_step[0]

        messages = [
            [{
            "role": "user",
            "content":  self.SUM.format(question, "\n".join(reasoning_step))
            }]
            ]
        # Generate the output with stop conditions
        temperature = random.choice(self.temperature_range)
        final_pred = self.generator.generate(messages, stop=["Doc", "#"], temperature=temperature, n=1)[0]
        reasoning_step.append(final_pred)
        return reasoning_step, bad_gen
    
    
    def run_item_adaptive(self, question, partial_answer, step, num, if_end=False):
        question = question + '\n' + partial_answer
        reasoning_step_list = []
        accumulated_output = ""
        system_prompt = self.P_INS
        gen_count = 0

        
        user_prompt = (
            f"#\nQuestion: {question}\n"
            + accumulated_output
        )
        messages = [
            [{
            "role": "system",
            "content": system_prompt
            },
            {
            "role": "user",
            "content":  user_prompt
            }]]

        # Generate the output with stop conditions
        temperature = random.choice(self.temperature_range)
        gen_out_list = self.generator.generate(messages, stop=["Doc", "#"], temperature=temperature, n=num)[0]
        gen_count += 1
        if type(gen_out_list) == str:
            gen_out_list = [gen_out_list]
        # Build the prompt with the question and any accumulated output            
        for gen_out in gen_out_list:
            # 在生成 follow up 之后进行后续判定
            while True:
                stop_condition = "Search: Yes."
                if stop_condition in gen_out:
                    gen_out = gen_out.split(stop_condition)[0] + stop_condition
                    # Extract the last follow-up question from the generated output
                    follow_up_questions = re.findall(self.FOLLOW_UP_PATTERN, gen_out)
                    if follow_up_questions:
                        accumulated_output += gen_out  # Accumulate output for continuity in next iteration
                        latest_follow_up = follow_up_questions[-1].replace("Follow up: ", "").strip()
                        # Perform retrieval based on the latest follow-up question
                        retrieval_result = self.retriever.search(latest_follow_up)
                        # Format and add retrieval results to accumulated_output
                        formatted_reference = self.format_reference(retrieval_result)
                        accumulated_output += formatted_reference+"\nIntermediate answer:"
                    else: 
                        accumulated_output_copy = accumulated_output
                        parts = accumulated_output.rsplit("Search: ", 1)
                        if accumulated_output == "": # 模型需要直接回答
                            retrieval_result = self.retriever.search(question)
                            # Format and add retrieval results to accumulated_output
                            formatted_reference = self.format_reference(retrieval_result)
                            accumulated_output += formatted_reference
                            accumulated_output += "\nSo the final answer is:"
                        else:
                            modified_last_part = "No."+"\nRetrieved documents:" + "\nIntermediate answer:"
                            accumulated_output = parts[0] + "Search: " + modified_last_part
                            if accumulated_output_copy == accumulated_output: # 陷入死循环了，采用直接回答终止
                                retrieval_result = self.retriever.search(question)
                                # Format and add retrieval results to accumulated_output
                                formatted_reference = self.format_reference(retrieval_result)
                                accumulated_output = ""
                                accumulated_output += formatted_reference
                                accumulated_output += "\nSo the final answer is:"
                        print(accumulated_output)
                    
                elif "So the final answer is:" in gen_out or "So the final answer is:" in accumulated_output:
                    # Final answer is generated, store and exit loop
                    accumulated_output += gen_out  # Accumulate output for continuity in next iteration
                    break
                else:
                    accumulated_output += gen_out
                    reasoning_step = self.extract_reasoning_steps(accumulated_output)
                    if (len(reasoning_step) == step-1 or gen_count==15) and if_end:
                        accumulated_output += "\nSo the final answer is:"
                    else:
                        accumulated_output += "\nFollow up:"
                # Check if reasoning count exceeds the max reasoning step limit
                reasoning_step = self.extract_reasoning_steps(accumulated_output)
                if len(reasoning_step)>=step:
                    break
                user_prompt = (
                    f"#\nQuestion: {question}\n"
                    + accumulated_output
                )
                messages = [
                    [{
                    "role": "system",
                    "content": system_prompt
                    },
                    {
                    "role": "user",
                    "content":  user_prompt
                    }]]
                gen_out = self.generator.generate(messages, stop=["Doc", "#"], temperature=temperature, n=1)[0]
                gen_count += 1
                if gen_count>15:
                    break
            reasoning_step_final = self.extract_reasoning_steps(accumulated_output)
            if len(reasoning_step_final) == 0:
                pass
            else:
                reasoning_step_list.append(reasoning_step_final[:step])
        return reasoning_step_list
    def update_values(self, score_list_per, alpha):
        """
        Update the values in score_list_per according to the formula:
        V(s_t) <- V(s_t) + alpha * (V(s_{t+1}) - V(s_t))
        
        Args:
            score_list_per (list[float]): List of V values for each time step.
            alpha (float): Update step size (learning rate).
        
        Returns:
            list[float]: Updated list of V values.
        """
        updated_scores = score_list_per.copy()  # Create a copy to avoid in-place modification
        update_step = len(score_list_per) - 1
        for t in range(len(score_list_per) - 1):
            delta =  score_list_per[t + 1] - score_list_per[t]
            if delta < 0.05:
                update_step = t + 1
                break

        # Update values from the end of the list to the start
        for t in range(update_step - 1, -1, -1):
            updated_scores[t] = updated_scores[t] + alpha * (updated_scores[t + 1] - updated_scores[t])
        return updated_scores[0]
    

    def prm_score(self, question, history_steps, reasoning_steps, alpha=0.5):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        system_prompt = self.P_INS
        user_prompt = (
            f"#\nQuestion: {question}\n"
        )
        tokenizer = self.prm.tokenizer
        prm_model = self.prm.model
        
        if isinstance(reasoning_steps, list):
            # if self.is_v100:
                score_list = []
                for reasoning_step in reasoning_steps:
                    if reasoning_step == []:
                        continue
                    # 构造 batch 输入
                    model_inputs_batch = []
                    for step_id in range(len(reasoning_step)):
                        model_input = [
                            {
                                "role": "user",
                                "content": system_prompt + '\n' + user_prompt
                            },
                            {
                                "role": "assistant",
                                "content": history_steps + "\n" + "\n".join(reasoning_step[:step_id+1])
                            }
                        ]
                        model_inputs_batch.append(model_input)

                    # 批量 tokenization
                    tokenized_inputs = tokenizer.apply_chat_template(
                        model_inputs_batch,
                        tokenize=True,
                        return_tensors="pt",
                        padding=True  # 处理不同长度的输入
                    ).to(prm_model.device)

                    # 批量推理
                    logits = prm_model(tokenized_inputs).logits[:, 0].tolist()

                    try:
                        score = self.update_values(logits, alpha=alpha)
                        score_list.append(sigmoid(score))
                        
                    except Exception as e:
                        print("Error:", e)

                return score_list
            # else:
                # score_list = []

                # # 构造所有推理步骤的批次输入，并记录每个 reasoning_step 的长度
                # model_inputs_batch = []
                # step_lengths = []  # 用来存储每个 reasoning_step 的长度

                # for reasoning_step in reasoning_steps:
                #     if reasoning_step == []:
                #         continue
                #     step_lengths.append(len(reasoning_step))  # 记录当前 reasoning_step 的长度
                #     for step_id in range(len(reasoning_step)):
                #         model_input = [
                #             {
                #                 "role": "user",
                #                 "content": system_prompt + '\n' + user_prompt
                #             },
                #             {
                #                 "role": "assistant",
                #                 "content": history_steps + "\n" + "\n".join(reasoning_step[:step_id+1])
                #             }
                #         ]
                #         model_inputs_batch.append(model_input)

                # # 批量 tokenization
                # tokenized_inputs = tokenizer.apply_chat_template(
                #     model_inputs_batch,
                #     tokenize=True,
                #     return_tensors="pt",
                #     padding=True  # 处理不同长度的输入
                # ).to(prm_model.device)

                # # 批量推理
                # logits = prm_model(tokenized_inputs).logits[:, 0].tolist()

                # # 处理每个 reasoning_step 对应的得分
                # logits_index = 0  # 用于跟踪当前 logits 在批量中的位置

                # for length in step_lengths:
                #     try:
                #         # 从 logits 中取出与当前 reasoning_step 对应的 logits
                #         logits_batch = logits[logits_index:logits_index + length]
                #         logits_index += length  # 更新 logits 索引

                #         # 使用 update_values 来计算得分
                #         score = self.update_values(logits_batch, alpha=alpha)  # input: list output: float 
                #         score_list.append(sigmoid(score))

                #     except Exception as e:
                #         print("Error:", e)

                # return score_list
        
        elif isinstance(reasoning_steps, str):
            model_input =   [
                                    {
                                        "role": "user",
                                        "content": system_prompt + '\n' + user_prompt
                                    },
                                    {
                                        "role": "assistant",
                                        "content": history_steps + "\n"+reasoning_steps
                                    }
                            ]
            model_inputs = tokenizer.apply_chat_template(
                        model_input,
                        tokenize=True, 
                        return_tensors="pt"
                    ).to(prm_model.device)
            score = sigmoid(prm_model(model_inputs).logits[0][0].item())
            return score


    def critic_exp_model(self, question, history_steps, current_step, furure_reasoning_steps, reward_score):
        is_openai = self.config["exp_framework"]=="openai"
        if is_openai:
            model_input = [[
                            {"role": "system", "content": self.EXP_SYS},
                            {"role": "user", "content": self.EXP.format(question, history_steps, current_step, furure_reasoning_steps, reward_score)}
                          ]]
            explanation = self.explaner.generate(model_input)[0]
        else:
            tokenizer = self.explaner.tokenizer
            explaner = self.explaner.model
            model_input = [
                            {"role": "system", "content": self.EXP_SYS},
                            {"role": "user", "content": self.EXP.format(question, history_steps, current_step, furure_reasoning_steps, reward_score)}
                        ]
            text = tokenizer.apply_chat_template(
                        model_input,
                        tokenize=False,
                        add_generation_prompt=True
                    )
            model_inputs = tokenizer([text], return_tensors="pt").to(explaner.device)
            generated_ids = explaner.generate(**model_inputs, max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            explanation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return explanation
    
    
    def save_item(self, item, save_path):
        """Save the dataset into the original format."""

        def convert_to_float(d):
            return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in d.items()}

        save_data = convert_to_float(item.to_dict())
        with open(save_path, "a") as f:
            f.write(json.dumps(save_data) + "\n")

    def correction_step(self, question, history_steps, current_step, reward_explanation, reward_score):
        # involving the generation step
        model_input = self.COR.format(question, history_steps, current_step, reward_score, reward_explanation)
        system_prompt = self.P_INS
        
        user_prompt = (
            f"#\nQuestion: {question}\n"
            + history_steps
        )
        messages = [
            [{
            "role": "system",
            "content": system_prompt
            },
            {
            "role": "user",
            "content":  user_prompt
            },
            {
            "role": "assistant",
            "content":  current_step
            },
            {
            "role": "user",
            "content":  model_input
            }
            ]]

        # Generate the output with stop conditions
        temperature = random.choice(self.temperature_range)
        correct_action = self.generator.generate(messages, stop=["Doc", "#"], temperature=temperature, n=1)[0]
        try:
            correct_action_dict = json.loads(correct_action)
        except:
            correct_action_dict = {"Action": "Action3"}
        """
        1. 提取动作
        2. 执行生成
        3. 提取生成结果
            1. 对于action One 和 Three 不需要考虑任何历史, 直接进行正常生成即可
            2. 对于action Two 需要考虑搜索步骤，之后进行生成
        4. 生成之后提取 So the final answer is: / Intermediate answer: 拼接到 reasoning 中
        """
        Reflection = f"Reflection: The previous reasoning step had the following issues: {reward_explanation}, therefore, I need to refine the reasoning step as follows:"
        try:
            if "Action1" in correct_action_dict["Action"] or "Action 1" in correct_action_dict["Action"]:
                new_query = correct_action_dict["Query"]
                search = correct_action_dict["Search"]
                if "yes" in search.lower():
                    new_reasoning_step = "Follow up: " + new_query + '\n' + "Search: " + "Yes.\n"
                    retrieval_result = self.retriever.search(new_query)
                    # Format and add retrieval results to accumulated_output
                    formatted_reference = self.format_reference(retrieval_result)
                    new_reasoning_step += "Retrieved documents:\n" + formatted_reference + "\nIntermediate answer:"
                elif "no" in search.lower():
                    new_reasoning_step = "Follow up: " + new_query + '\n' + "Search: " + "No.\n"
                    # Format and add retrieval results to accumulated_output
                    new_reasoning_step += "Intermediate answer:"
                else:
                    print(correct_action)
                    raise TypeError
            elif "Action2" in correct_action_dict["Action"] or "Action 2" in correct_action_dict["Action"]:
                old_query = correct_action_dict["Query"]
                search = correct_action_dict["Search"]
                if "yes" in search.lower():
                    new_reasoning_step = "Follow up: " + old_query + '\n' + "Search: " + "Yes.\n"
                    retrieval_result = self.retriever.search(old_query)
                    # Format and add retrieval results to accumulated_output
                    formatted_reference = self.format_reference(retrieval_result)
                    new_reasoning_step += "Retrieved documents:\n" + formatted_reference + "\nIntermediate answer:"
                elif "no" in search.lower():
                    new_reasoning_step = "Follow up: " + old_query + '\n' + "Search: " + "No.\n"
                    # Format and add retrieval results to accumulated_output
                    new_reasoning_step += "Intermediate answer: "
                else:
                    print(correct_action)
                    raise TypeError
            elif "Action3" in correct_action_dict["Action"] or "Action 3" in correct_action_dict["Action"]:
                new_reasoning_step = "\nSo the final answer is: "
            else:
                print(correct_action)
                new_reasoning_step = "\nSo the final answer is: "
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
            new_reasoning_step = current_step

        user_prompt = (
            f"#\nQuestion: {question}\n"
            + f"{history_steps}\n"
            # + '\n'
            # + Reflection
            + new_reasoning_step
        )
        messages = [
            [{
            "role": "system",
            "content": system_prompt
            },
            {
            "role": "user",
            "content":  user_prompt
            }]]

        # Generate the output with stop conditions
        temperature = random.choice(self.temperature_range)
        gen_out = self.generator.generate(messages, stop=["Doc", "#"], temperature=temperature)[0]
        new_reasoning_step += gen_out
        if "Follow up" in gen_out and "So the final answer is:" in new_reasoning_step: # add fix due to instruction-following error 
            new_reasoning_step = current_step

        new_score = self.prm_score(question, history_steps, new_reasoning_step)

        reasoning_step_final = self.extract_reasoning_steps(new_reasoning_step)
        if reasoning_step_final == []:
            reasoning_step_final = [current_step]
            

        return reasoning_step_final, new_score, correct_action_dict
    

    def check_correctness(self, expected_answer, generated_response):
        pred = generated_response.lower()
        if generated_response=="":
            return False
        # If expected_answer is a list, check each element
        if isinstance(expected_answer, list):
            for answer in expected_answer:
                answer = answer.lower()
                original_score = self.metric.token_level_scores(pred, answer)["f1"]
                if original_score > 0.5 or answer in pred:
                    return True
        else:
            expected_answer = expected_answer.lower()
            original_score = self.metric.token_level_scores(pred, expected_answer)["f1"]
            return original_score > 0.5 or expected_answer in pred
        
        return False
    
    @torch.inference_mode(mode=True)
    def run_item(self, item, step, num, beta=0.5, file_path="./hotpotqa/exp_data/exp_data.jsonl"):
        question = item.question
        all_retrieval_result = []
        iter_count = 0
        accumulated_output = []
        data_dict_list = []
        all_reasoning_steps = []
        while True:
            if len(accumulated_output)>3:
                step = 1
            if iter_count == self.max_iter-1 or len(accumulated_output) == self.max_iter - 1:
                if_end = True
            else:
                if_end = False

            
            reasoning_step_list = self.run_item_adaptive(question, "\n".join(accumulated_output), step, num, if_end=if_end)
            if reasoning_step_list == []:
                item.update_output("pred", "I don't know the answer.")
                self.save_item(save_path=self.config["save_dir"]+"/item_immediate.jsonl" , item=item)
                print("------zero output----")
                return
            score_list = self.prm_score(question, "\n".join(accumulated_output), reasoning_step_list)

            all_reasoning_steps.append({f"Step: {iter_count}":{"accumulated_output":accumulated_output, "reasoning_step_list":reasoning_step_list, "score_list":score_list}})
            max_idx = score_list.index(max(score_list))
            if max(score_list)<beta:
                print("start explanation critic")

                explanation = self.critic_exp_model(question,"\n".join(accumulated_output), reasoning_step_list[max_idx][0], "\n".join(reasoning_step_list[max_idx][1:]) if len(reasoning_step_list[max_idx])>1 else "", max(score_list))
                all_reasoning_steps[-1][f"Step: {iter_count}"]["explanation"] = explanation
                
                new_reasoning_step, new_score, action_dict = self.correction_step(item.question, "\n".join(accumulated_output), reasoning_step_list[max_idx][0], explanation, max(score_list))
                all_reasoning_steps[-1][f"Step: {iter_count}"]["new_reasoning_step"] = new_reasoning_step
                all_reasoning_steps[-1][f"Step: {iter_count}"]["new_score"] = new_score
                all_reasoning_steps[-1][f"Step: {iter_count}"]["action_dict"] = action_dict
                accumulated_output += new_reasoning_step
                orig_score = max(score_list)
                if new_score > orig_score:
                    data_dict = {"question":item.question, "explanation":explanation, "history_step":"\n".join(accumulated_output), "current_step": reasoning_step_list[max_idx][0], "future_step": "\n".join(reasoning_step_list[max_idx][1:]) if len(reasoning_step_list[max_idx])>1 else "", "orig_score":orig_score, "new_score":new_score, "label":1}
                else:
                    data_dict = {"question":item.question, "explanation":explanation, "history_step":"\n".join(accumulated_output), "current_step": reasoning_step_list[max_idx][0], "future_step": "\n".join(reasoning_step_list[max_idx][1:]) if len(reasoning_step_list[max_idx])>1 else "", "orig_score":orig_score, "new_score":new_score, "label":0}
                data_dict_list.append(data_dict)
            else:
                new_reasoning_step = reasoning_step_list[max_idx][0]
                accumulated_output.append(new_reasoning_step) # 只取当前 step

                all_reasoning_steps[-1][f"Step: {iter_count}"]["new_reasoning_step"] = new_reasoning_step
                all_reasoning_steps[-1][f"Step: {iter_count}"]["new_score"] = None
                all_reasoning_steps[-1][f"Step: {iter_count}"]["action_dict"] = None

            if "So the final answer is:" in accumulated_output[-1]:
                break
            iter_count += 1
            if iter_count >= self.max_iter or len(accumulated_output) >= self.max_iter or "So the final answer is:" in new_reasoning_step:
                break
        if "So the final answer is:" in accumulated_output[-1]:
            pred = accumulated_output[-1].split("So the final answer is:")[-1].strip()
        else:
            pred = ""
            for i in range(len(accumulated_output)-1, -1, -1):
                try:
                    pred += accumulated_output[i].split("Intermediate answer:")[-1].strip()
                except:
                    print("No Intermediate answer found in the response.")
                    print(accumulated_output[i])
                    continue
        pred = pred.lower()
        item.update_output("pred", pred)
        item.update_output("reasoning_steps", accumulated_output)
        # Store retrieval results for reference
        item.update_output("retrieval_result", all_retrieval_result)
        item.update_output("all_reasoning_steps", all_reasoning_steps)
        if self.check_correctness(item.golden_answers, item.pred):
            with open(file_path, "a") as f:
                for exp_data in data_dict_list:
                    f.write(json.dumps(exp_data) + "\n")
            print("save exp data!")
        self.save_item(save_path=self.config["save_dir"]+"/item_immediate.jsonl" , item=item)

    def extract_reasoning_steps(self, response):
        # Define regex pattern to capture each reasoning step
        pattern = re.compile(
            r"(Follow up:.*?\n"
            r"Search: (Yes|No).*?\n"
            r"(Retrieved documents:.*?\n)?"
            r"Intermediate answer:.*?)(?=\nFollow up:|So the final answer is:|$)|"
            r"(Retrieved documents:.*?\nSo the final answer is:.*?$)|"  # New pattern for final answer with retrieved docs
            r"(So the final answer is:.*?$)",  # Pattern for final answer only
            re.DOTALL
        )
        
        # Extract all reasoning steps
        steps = pattern.findall(response)

        # Clean up the extracted steps and store them in a list
        reasoning_steps = [step[0] if step[0] else (step[3] if step[3] else step[4]) for step in steps]

        # Display each reasoning step
        return reasoning_steps


    def _remove_duplicate_doc(self, docs):
        assert all(["id" in doc for doc in docs])
        new_doc_list = []
        exist_ids = []
        for doc in docs:
            doc_id = doc["id"]
            if doc_id not in exist_ids:
                exist_ids.append(doc_id)
                new_doc_list.append(doc)
        return new_doc_list
    
    def summary(self, item):
        question = item.question
        try:
            if hasattr(item, 'states'):
                reasoning_steps = item.states
            else:    
                reasoning_steps = item.reasoning_steps
        except:
            reasoning_steps = []
        messages = [
            [{
            "role": "user",
            "content":  self.SUM.format(question, "\n".join(reasoning_steps))
            }]
            ]

        # Generate the output with stop conditions
        temperature = random.choice(self.temperature_range)
        final_pred = self.generator.generate(messages, stop=["Doc", "#"], temperature=temperature, n=1)[0]
        if final_pred=="":
            print(messages)
        item.update_output("old_pred", item.pred)    
        item.update_output("pred", final_pred)
        time.sleep(0.1)


    def run(self, dataset, do_eval=True, pred_process_fun=rag_reasoning_pred_parse, file_path="./hotpotqa/exp_data/exp_data.jsonl",):
        for idx, item in enumerate(tqdm(dataset, desc="Inference: ")):
            if "gpt" not in self.config.generator_model:
                beta = 0.25
            else:
                beta = 0.5
            self.run_item(item, step=3, num=3, file_path=file_path, beta=beta)
            self.summary(item)
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun, )
        return dataset
    def run_exp_data(self, dataset, do_eval=True, pred_process_fun=rag_reasoning_pred_parse):
        for item in tqdm(dataset, desc="Inference: "):
            self.run_item_exp_data(item, step=3, num=1, file_path="./hotpotqa/exp_data/exp_data.jsonl")
        return dataset