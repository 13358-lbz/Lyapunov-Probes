# extract hidden layers from model for each sentence in a training dataset
# this hidden layers are then used to train a classifier

import json
import logging
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # specify the GPUs to use

from pathlib import Path

import torch
import typer
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import trange
from transformers import BitsAndBytesConfig
from torch.nn.functional import cosine_similarity

from self_knowledge.arch import get_model

accelerator = Accelerator()


def save_hidden_layer(
    model_name: str = "/data/bozhiluan/factual-confidence-of-llms-main/tiiuae/falcon-7b-instruct",
    dataset_path: str = "results_kn_popqa/tiiuae/falcon-7b-instruct/popqa_sf_None.csv",
    continue_from: str = None,
    log_path: str = "logs",
    hidden_layer_index: int = -1,
    save_file: str = "models/tiiuae/falcon-7b-instruct/lama3_hlayer_24.pt",
    batch_size: int = 1,
    save_freq: int = 1000,
):
    Path(save_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=f"{log_path}/log_hidden.txt", level=logging.DEBUG)
    # model, tokenizer = get_model(model_name, load_in_4bit=True, device_map="balanced")
    # 创建量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # 使用 float16 以获得更好的性能
        bnb_4bit_use_double_quant=True,  # 使用双重量化节省更多内存
        bnb_4bit_quant_type="nf4"  # 使用 NF4 量化类型
    )
    
    # 移除 device_map 以避免与 accelerate 冲突
    model, tokenizer = get_model(
        model_name, 
        quantization_config=quantization_config
    )
    model, tokenizer = accelerator.prepare(model, tokenizer)
    if ".csv" in dataset_path:
        dataset = load_dataset("csv", data_files=dataset_path)["train"]
        tr_dataset = dataset.train_test_split(test_size=0.2, seed=42)["train"]
    else:
        tr_dataset = load_from_disk(dataset_path)

    # as accelerate is going back and forth between GPU and CPU, we separate them into two lists
    hidden_layer = []  # on GPU or CPU, smaller
    out = []  # only CPU, all data
    starting_idx = 0
    if continue_from is not None:
        out = torch.load(continue_from)
        starting_idx = len(out) // save_freq
    for i in trange(starting_idx, len(tr_dataset) // save_freq):
        restricted_dataset = tr_dataset.select(
            range(i * save_freq, (i + 1) * save_freq, 1)
        )
        tr_dataloader = DataLoader(restricted_dataset, batch_size=batch_size)
        for batch in tr_dataloader:
            _tok_batch = tokenizer(batch["text"], padding=True, return_tensors="pt").to(
                accelerator.device
            )

            o = model(**_tok_batch, return_dict=True, output_hidden_states=True)
            orig_hidden_states = o.hidden_states[0]
            for i in range(o.hidden_states[hidden_layer_index].shape[0]):
                hidden_layer.append(
                    {
                        "hidden": o.hidden_states[hidden_layer_index][i, -1:, :]
                        .view(-1)
                        .cpu(),
                        "uuid": batch["uuid"][i],
                        "is_factual": batch[
                            (
                                "generation_correct"
                                if "generation_correct" in batch.keys()
                                else "is_factual"
                            )
                        ][i],
                    }
                )
            # print hidden_layer
            print("is_factual", hidden_layer[-1]["is_factual"])

            # lbz add
            # 在生成_tok_batch之后添加扰动
            # _tok_batch = tokenizer(batch["text"], padding=True, return_tensors="pt").to(accelerator.device)

            # ------------ 添加扰动 ------------
            if batch["generation_correct"].item() is True:
                # 如果是事实性生成，则进行扰动
                print("generation_correct is True. Perturbing input text for factual generation...")

                perturb_times = 3  # 扰动次数
                for _ in range(perturb_times):
                    # 克隆input_ids以避免修改原始数据
                    perturbed_input_ids = _tok_batch["input_ids"].clone()
                    batch_size, seq_len = perturbed_input_ids.shape

                    # 扰动概率（例如30%的token被扰动）
                    # perturb_prob = 0.3
                    perturb_prob = 0.0

                    for i in range(batch_size):
                        # 只对有效token（attention_mask=1的位置）进行扰动
                        valid_positions = torch.where(_tok_batch["attention_mask"][i] == 1)[0]
                        if len(valid_positions) == 0:
                            continue
                        
                        # 随机选择要扰动的位置
                        num_perturb = max(0, int(len(valid_positions) * perturb_prob))  # 至少扰动1个token
                        perturb_positions = torch.randperm(len(valid_positions))[:num_perturb]
                        perturb_indices = valid_positions[perturb_positions]
                        
                        # 对选中的位置替换为随机有效token（确保在词表范围内）
                        for pos in perturb_indices:
                            # 生成一个随机token ID（排除特殊token如<unk>可进一步优化）
                            random_token_id = torch.randint(
                                low=0, 
                                high=tokenizer.vocab_size,  # 词表大小，确保ID有效
                                size=(), 
                                device=perturbed_input_ids.device
                            )
                            perturbed_input_ids[i, pos] = random_token_id

                    # decode一下
                    decode_text = tokenizer.batch_decode(
                        perturbed_input_ids, skip_special_tokens=True
                    )
                    print(f"Original text: {batch['text']}")
                    print(f"Perturbed text: {decode_text}")

                    

                    _tok_batch_copy = _tok_batch.copy()
                    _tok_batch_copy["input_ids"] = perturbed_input_ids

                    o = model(**_tok_batch_copy, return_dict=True, output_hidden_states=True)

                    # 假设 hidden_states[0] 形状为 [1, 7, 4544]
                    pert_hidden_states = o.hidden_states[0]  # 第一层隐藏状态

                    # 原始文本和扰动文本的第一层隐藏状态聚合特征
                    orig_emb = torch.mean(orig_hidden_states, dim=1)  # [1, 4544]
                    pert_emb = torch.mean(pert_hidden_states, dim=1)  # [1, 4544]

                    # 2. 确保张量是2维的（移除多余的维度）
                    # 如果你使用的是单样本（batch_size=1），可能需要挤压第一个维度
                    orig_emb = orig_emb.squeeze(0)  # 变为 [4544]
                    pert_emb = pert_emb.squeeze(0)  # 变为 [4544]

                    # 计算余弦相似度（范围 [-1, 1]）
                    sim = cosine_similarity(orig_emb, pert_emb, dim=-1).item()

                    # 转换为扰动距离（范围 [0, 2]）
                    perturb_distance = 1 - sim

                    for i in range(o.hidden_states[hidden_layer_index].shape[0]):
                        hidden_layer.append(
                            {
                                "hidden": o.hidden_states[hidden_layer_index][i, -1:, :]
                                .view(-1)
                                .cpu(),
                                "uuid": batch["uuid"][i],
                                "is_factual": torch.tensor(1-perturb_distance),  # 假设扰动后仍然是事实性生成
                            }
                        )
                    # print hidden_layer
                    print("is_factual", hidden_layer[-1]["is_factual"])
                    print("Perturbed input processed.")
                    # -----------------------------------
                # -----------------------------------

            # 继续模型推理
            # o = model(**_tok_batch, return_dict=True, output_hidden_states=True)
            # lbz add end

        # gather objects from all processes
        hidden_layer = accelerator.gather_for_metrics(hidden_layer)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.save(
                out + hidden_layer,
                save_file,
            )
        out = out + hidden_layer
        hidden_layer = []


def cleanup_dataset(
    hidden_path: str = "./models/hidden_layers/",
    out_name: str = "fal_7b_hidden.pt",
):
    hidden = []
    j = 0
    for file in Path(hidden_path).glob("*.pt"):
        if out_name not in str(file):
            hidden_layer = torch.load(file)
            uuid = json.loads(
                file.parent.joinpath(
                    file.name.split("hidden")[0] + "uuids.json"
                ).read_text()
            )
            for i in range(len(hidden_layer)):
                hidden.append(
                    {
                        "uuid": uuid["uuids"][i],
                        "hidden": hidden_layer[i],
                        "is_factual": uuid["factuality"][i],
                    }
                )
        j += 1
        if j % 10 == 0 or (j == len(os.listdir(hidden_path)) - 1):
            torch.save(hidden, Path(hidden_path) / out_name)


if __name__ == "__main__":
    typer.run(save_hidden_layer)
