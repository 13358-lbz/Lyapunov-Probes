import logging
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import tqdm
import typer
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)

import json
import re

from self_knowledge.arch import get_model
from self_knowledge.data_gen.paraphrase.nli import NLI

accelerator = Accelerator()


def clean_and_parse_json(ans_str):
    # 1. 移除所有换行符和多余空格
    ans_str = re.sub(r'\s+', ' ', ans_str).strip()
    
    # 2. 处理numpy数组标记
    ans_str = re.sub(r',\s*dtype=object', '', ans_str)
    ans_str = re.sub(r'array\(\s*\[(.*?)\]\s*\)', r'[\1]', ans_str, flags=re.DOTALL)
    
    # 3. 处理引号问题
    # 确保键名用双引号包裹
    ans_str = re.sub(r"(\s*)(\w+)(\s*):", r'\1"\2"\3:', ans_str)
    # 将单引号字符串转换为双引号
    ans_str = re.sub(r"'([^']*)'", r'"\1"', ans_str)
    
    # 4. 尝试解析
    try:
        return json.loads(ans_str)
    except json.JSONDecodeError as e:
        # print(f"解析错误: {e}")
        # print(f"处理后的字符串: {ans_str}")
        return None
    
def slot_fill(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    sentences: Dict[str, str],
    accelerator: Accelerator,
    is_popqa: bool = False,
) -> Tuple[bool, List[str]]:
    """get a batch of sentences from Trex or PopQA
    generate an amount of words greedily that corresponds to the possible answer
    check and label if answer is correct
    Parameters
    ----------
    model : torch.nn.Module
    tokenizer : transformers.PreTrainedTokenizerFast
    sentences : Dict{text: str, obj_label: str}
    accelerator : accelerate.Accelerator
    is_popqa : bool
        if True, we use the PopQA dataset, else we use Trex
    Returns
    -------
    success : list of bool
    """
    if is_popqa:
        # ans = "obj"
        ans = "answer"
        question = "text"
        # question = "question"
        nli = None
    else:
        ans = "obj_label"
        question = "prompt"
        nli = NLI()

    max_new_tokens = 20

    # popqa
    question_prompt = sentences[question]
    
    # # triviaqa
    # # 对于每一条，在Question: 前面截断，保留set_length个字符
    # set_length = 2000
    # for i in range(len(question_prompt)):
    #     question_text = ""
    #     context = ""
    #     if "Question:" in question_prompt[i]:
    #         question_text = question_prompt[i].split("Question:")[-1].split("\nAnswer:")[0]
    #         context = question_prompt[i].split("Question:")[0].replace("Context:", "").strip()
    #     if len(context) > set_length:
    #         context = context[:set_length] + '}'
    #     question_prompt[i] = f"Context: {context} \nQuestion: {question_text}\nAnswer with a single word or phrase.\nAnswer:"
    
    # NQ
    for i in range(len(question_prompt)):
        question_prompt[i] = question_prompt[i] + "? \nAnswer:"

    # # hotpotqa
    # # question_prompt = [sentences["context"][i] + " " + sentences[question][i] for i in range(len(sentences[question]))]
    # question_prompt = []
    # for i in range(len(sentences[question])):
    #     context = sentences["context"][i]
    #     question_text = sentences[question][i]
    #     context.replace("[PAR]", " ")
    #     context.replace("[TLE]", " ")
    #     question_prompt.append(f"Context: {context}\nQuestion: {question_text}\nAnswer:")
    # answers = sentences[ans]

    input_tok = tokenizer(
        question_prompt, padding=True, truncation=True, return_tensors="pt"
    ).to(accelerator.device)
    gen_o = model.generate(
        **input_tok, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id
    )
    logging.debug(f"input: {question_prompt}")
    generated = tokenizer.batch_decode(gen_o, skip_special_tokens=True)
    # 去除prompt部分
    # generated = [gen.split("Answer:")[-1].strip() for gen in generated]
    logging.debug(f"generated: {generated}")
    logging.debug(f"expected: {sentences[ans]}")

    # in the popqa case, we check all proposed options
    if is_popqa:
        # is_factual = [sentences[ans][i] in _gen for i, _gen in enumerate(generated)]

        # popqa
        is_factual = []
        for i, _gen in enumerate(generated):
            # 检查 sentences[ans][i] 是否在生成的文本 _gen 中
            if sentences[ans][i] in _gen:
                is_factual.append(True)
            else:
                is_factual.append(False)

        # # triviaqa
        # is_factual = []
        # for i, _gen in enumerate(generated):
        #     # 检查 sentences[ans][i] 是否在生成的文本 _gen 中
        #     # sentences[ans][i]="{'aliases': array(['Four', 'four', '4'], dtype=object), 'normalized_aliases': array(['four', '4'], dtype=object), 'matched_wiki_entity_name': '', 'normalized_matched_wiki_entity_name': '', 'normalized_value': 'four', 'type': 'Numerical', 'value': 'Four'}"
        #     # 处理原始答案字符串
        #     ans_str = sentences[ans][i]

        #     answer = clean_and_parse_json(ans_str)
        #     if answer is None:
        #         is_factual.append(False)
        #         continue
        #     exact_value = answer["value"]  # 精确答案（如 "Four"）
        #     aliases = answer["aliases"]    # 同义变体（如 ["Four", "four", "4"]）
        #     normalized_value = answer["normalized_value"]  # 标准化答案（如 "four"）
        #     normalized_aliases = answer["normalized_aliases"]  # 标准化变体（如 ["four", "4"]）

        #     # 合并所有可能的正确答案（去重）
        #     all_correct_answers = list(set([exact_value] + aliases + [normalized_value] + normalized_aliases))
        #     # print("所有可能的正确答案：", all_correct_answers)

        #     if any(answer in _gen for answer in all_correct_answers):
        #         is_factual.append(True)
        #     else:
        #         is_factual.append(False)

        # # hotpotqa
        # is_factual = []
        # for i, _gen in enumerate(generated):
        #     # 检查 sentences[ans][i] 是否在生成的文本 _gen 中
        #     answer = sentences[ans][i].strip("[]").strip("\'").strip('"')
        #     if answer in _gen:
        #         is_factual.append(True)
        #     else:
        #         is_factual.append(False)

    else:
        _decoded = tokenizer.batch_decode(gen_o)
        # concat prompt and answer
        _truth = [
            f"{sentences[question][i]} {sentences[ans][i]}"
            for i in range(len(sentences[ans]))
        ]
        success = nli.check_equivalence(_truth, _decoded)
        failure = []
        for i, s in enumerate(generated):
            failure.append(False)
            if not success[i]:
                for word in s.split(" "):
                    if word in sentences["verified_false_facts"][i]:
                        failure[i] = True
                        break
        # three table into one table with 3 classes
        is_factual = [
            "factual" if s else "non-factual" if f else "unknown"
            for s, f in zip(success, failure)
        ]
    logging.debug(f"is_factual: {is_factual}")
    return is_factual, generated


def evaluate_all_slot_filling(
    model, tokenizer, dataloader, accelerator, is_popqa=False
) -> None:
    """evaluate all slot filling tasks"""
    out = {
        key: []
        for key in ["generated", "generation_correct", *dataloader.dataset.column_names]
    }
    for batch in tqdm.tqdm(dataloader):
        sc, gen = slot_fill(model, tokenizer, batch, accelerator, is_popqa=is_popqa)
        out["generated"].extend(gen)
        out["generation_correct"].extend(sc)
        for key in dataloader.dataset.column_names:
            out[key].extend(batch[key])
            # if needed, to cpu
            if isinstance(out[key][0], torch.Tensor):
                out[key][-1] = out[key][-1].cpu()
    return out


def log_to_pandas_dataframe(logpath: str = "logs/slot_filling_2/", save: bool = True):
    data = {"input": [], "text": [], "expected": [], "is_factual": []}
    for file in glob.glob(f"{logpath}*.log"):
        with open(file, "r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if "input" in lines[i]:
                    data["input"].extend(
                        [
                            text.replace("'", "")
                            .replace('"', "")
                            .replace("[", "")
                            .replace("]", "")
                            .replace("\n", "")
                            for text in lines[i].split("input: ")[1].split("',")
                        ]
                    )
                elif "generated" in lines[i]:
                    data["text"].extend(
                        [
                            text.replace("'", "")
                            .replace('"', "")
                            .replace("[", "")
                            .replace("]", "")
                            .replace("<|endoftext|>", "")
                            .replace("\n", "")
                            for text in lines[i].split("generated: ")[1].split("',")
                        ]
                    )
                elif "expected" in lines[i]:
                    data["expected"].extend(
                        [
                            text.replace("'", "")
                            .replace('"', "")
                            .replace("[", "")
                            .replace("]", "")
                            .replace("\n", "")
                            for text in lines[i].split("expected: ")[1].split(",")
                        ]
                    )
                elif "success" in lines[i]:
                    data["is_factual"].extend(
                        [
                            text.replace("[", "").replace("]", "").replace("\n", "")
                            == "True"
                            for text in lines[i].split("success: ")[1].split(",")
                        ]
                    )
    if save:
        pd.DataFrame(data).to_csv(f"{logpath}/slot_filling.csv")
    return data


def lama_gen(
    outpath="data/lama_sf.csv",
    log_path="logs/improved_sf",
    dataset_path="data/wikidata_trex",
    model_name="tiiuae/falcon-7b-instruct",
    batch_size=2,
    accelerator=None,
):
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)
    Path(log_path).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f"{log_path}/gpu_{str(accelerator.device).split(':')[-1]}.log",
    )
    model, tokenizer = get_model(model_name=model_name)
    # load all jsonl in dataset_path
    tr_dataset = pd.DataFrame()
    for file in glob.glob(f"{dataset_path}/*.jsonl"):
        tr_dataset = pd.concat([tr_dataset, pd.read_json(file, lines=True)])
    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size)
    model, tokenizer, tr_dataloader = accelerator.prepare(
        model, tokenizer, tr_dataloader
    )
    sf = torch.stack(
        evaluate_all_slot_filling(
            model, tokenizer, tr_dataloader, is_popqa=False, accelerator=accelerator
        )
    )
    sf.to_csv(f"{outpath}/lama_sf_{accelerator.device.index}.csv")


def popqa_gen(
    log_path="logs/pop_slot_filling",
    data_path="data/trivia_qa/train_rand10k_prepared.csv",
    output_path="results_kn/",
    model_name="download_models/Meta-Llama-3-8B-Instruct",
    output_name="popqa_sf",
    no_accelerator=True,
    batch_size: int = 8,
    train_split=True,
):
    Path(log_path).mkdir(exist_ok=True, parents=True)
    Path(output_path).mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.DEBUG,
        filename=f"{log_path}/gpu_{str(accelerator.device).split(':')[-1]}.log",
    )
    dataset = load_dataset("csv", data_files=data_path, delimiter=",")["train"].shuffle(
        seed=42
    )
    print(batch_size)
    tr_dataloader = DataLoader(dataset, batch_size=batch_size)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model, tokenizer = get_model(
        model_name=model_name,
        quantization_config=bnb_config,
        device_map="balanced" if no_accelerator else None,
        # load_in_4bit=True,
        trust_remote_code=True  # 必须添加这行
    )
    if not no_accelerator:
        model, tokenizer, tr_dataloader = accelerator.prepare(
            model, tokenizer, tr_dataloader
        )
    sf = evaluate_all_slot_filling(
        model, tokenizer, tr_dataloader, is_popqa=True, accelerator=accelerator
    )
    # pd.DataFrame(sf).to_csv(f"{output_path}/popqa_sf_{accelerator.device.index}.csv")
    pd.DataFrame(sf).to_csv(f"{output_path}/{output_name}_{accelerator.device.index}.csv")


if __name__ == "__main__":
    typer.run(popqa_gen)
