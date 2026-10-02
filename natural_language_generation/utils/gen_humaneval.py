import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "peft", "src"))
from peft import PeftModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HumanEval/MBPP predictions with a SWIFT adapter.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--data_path", default="pissa-dataset")
    parser.add_argument("--sub_task", default="python")
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--output_file", default="python_response.jsonl")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    return parser.parse_args()


def batches(values, batch_size):
    if batch_size < 1:
        raise ValueError("--batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def main():
    args = parse_args()
    if not os.path.isdir(args.adapter_path):
        raise FileNotFoundError(f"SWIFT adapter directory not found: {args.adapter_path}")

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype or "auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = PeftModel.from_pretrained(model, args.adapter_path, device_map="auto")
    model.eval()

    generation_config = GenerationConfig(
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        stop_strings=["### Instruction:", "### Input:", "</s>"],
    )
    dataset = load_dataset(
        args.data_path,
        data_dir=args.sub_task,
        split=args.dataset_split,
    )
    required_columns = {"instruction", "output", "type"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    parent = os.path.dirname(os.path.abspath(args.output_file))
    os.makedirs(parent, exist_ok=True)
    queries = dataset["instruction"]
    answers = dataset["output"]
    task_types = dataset["type"]
    total_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    with open(args.output_file, "w", encoding="utf-8") as output_handle:
        iterator = zip(
            batches(queries, args.batch_size),
            batches(answers, args.batch_size),
            batches(task_types, args.batch_size),
        )
        for batch_queries, batch_answers, batch_tasks in tqdm(iterator, total=total_batches):
            inputs = tokenizer(batch_queries, return_tensors="pt", padding=True)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                sequences = model.generate(
                    **inputs,
                    tokenizer=tokenizer,
                    pad_token_id=tokenizer.pad_token_id,
                    generation_config=generation_config,
                )
            generated = sequences[:, inputs["input_ids"].shape[1] :]
            completions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for query, completion, answer, task in zip(
                batch_queries, completions, batch_answers, batch_tasks
            ):
                json.dump(
                    {"type": task, "query": query, "output": completion.strip(), "answer": answer},
                    output_handle,
                    ensure_ascii=False,
                )
                output_handle.write("\n")
                output_handle.flush()


if __name__ == "__main__":
    main()
