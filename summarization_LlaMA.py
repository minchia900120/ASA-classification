# -- coding:utf-8 --
import argparse
import json
import logging
from tqdm import tqdm
from huggingface_hub import login

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference on a summarization task.")
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=None,
        help="Token to login Hugging Face.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A json file containing the testing data."
    )
    # tokenizer
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--truncation",
        type=bool,
        default=True,
        help="Sequences longer than max_length will be truncated.",
    )
    parser.add_argument(
        "--return_tensors", 
        type=str, 
        choices=['pt', 'tf', 'np'],
        default="pt",
        help="Return tensors in PyTorch (pt), TensorFlow (tf), or NumPy (np) format.",
    )
    # model.generate
    parser.add_argument(
        "--max_new_tokens", 
        type=int,
        default=512,
        help="The maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--do_sample", 
        type=bool, 
        default=True, 
        help="Whether or not to use sampling; use greedy decoding if False.",
    )
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=50, 
        help="The number of highest probability vocabulary tokens to keep for top-k filtering.",
    )
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.92, 
        help="The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.",
    )
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.7, 
        help="The value used to modulate the next token probabilities.",
    )
    parser.add_argument(
        '--num_beams', 
        type=int, 
        default=1, 
        help="Number of beams for beam search. 1 means no beam search.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final result.",
    )
    args = parser.parse_args()
    return args

args = parse_args()

# 創建 logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)   # 設置日誌級別為 INFO
console_handler = logging.StreamHandler()   # 設置控制台輸出
console_handler.setLevel(logging.INFO)
# 定義 Formatter 並添加到處理器
formatter = logging.Formatter(
    '[%(asctime)s | %(filename)s | %(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)  # 添加處理器到 logger

# 用您在 Hugging Face 上生成的访问令牌进行登录
if args.huggingface_token is not None:
    login(args.huggingface_token)

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

# 如果有GPU并支持fp16，使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device: " + str(device))
model.to(device)

model.eval()

with open(args.test_file, "r", encoding="utf-8") as f:
    text = json.load(f)

result = []
for i in tqdm(range(len(text))):
    #text_i = "Please summarize the text: " + text[i]['input']
    text_i = "You are given information from the patient's medical record. Summarize this information, making sure to include the most important positive clinical findings." + text[i]['input']
    inputs = tokenizer(
        text_i,
        return_tensors=args.return_tensors,
        truncation=args.truncation,
        max_length=args.max_length
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        num_beams=args.num_beams
    )
    # 从 inputs 中提取 input_ids，并获取其形状
    input_ids = inputs['input_ids']
    new_outputs = outputs[:, input_ids.shape[-1]:]  # 只获取生成的新内容
    dict = {"instruction": text[i]['instruction'],
            "input": text[i]['input'],
            "summary": tokenizer.decode(new_outputs[0], skip_special_tokens=True),
            "output": text[i]['output']}
    result.append(dict)

f = open(args.output_dir, "w", encoding="utf-8")
json.dump(result, f, ensure_ascii=False)
logger.info("Done!")
