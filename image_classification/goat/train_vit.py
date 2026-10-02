import os
import argparse
import logging
import math

import pandas as pd
import torch
import torch.distributed as dist
import tqdm
import transformers
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import AutoProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Lambda,
    Normalize,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import peft
import peta
from peta.utils import TitledLog

log = logging.getLogger(__name__)
# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"

torch.set_float32_matmul_precision("medium")

def save_csv(data, out_path):
    # save excel
    columns = sorted(list(data.keys()))
    df = pd.DataFrame(data, index=[0]).reindex(columns=columns)
    os.makedirs(out_path, exist_ok=True)
    xlsx_path = os.path.join(out_path, 'results.csv')

    if os.path.exists(xlsx_path):
        previous = pd.read_csv(xlsx_path, index_col=0)
        df = pd.concat([previous, df])

    df.to_csv(xlsx_path, index=True)

def split_dataset(dataset, rank, world_size):
    total_size = len(dataset)
    per_process_size = math.ceil(total_size / world_size)
    start_index = rank * per_process_size
    end_index = min(start_index + per_process_size, total_size)
    subset = torch.utils.data.Subset(dataset, list(range(start_index, end_index)))
    return subset

def gather_from_all_processes(data):
    local_data = data.detach().cpu().tolist()
    gathered_data = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_data, local_data)
    return [item for sublist in gathered_data for item in sublist]

class CustomCallback(TrainerCallback):

    def __init__(self, trainer, test_dataset, args, **kwargs) -> None:
        super().__init__()
        self.trainer = trainer
        self.test_dataset = test_dataset
        self.local_rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        if self.world_size > 1:
            self.test_dataset = split_dataset(test_dataset, self.local_rank, self.world_size)
        self.args = args

    def on_epoch_end(self, args, state, control, **kwargs):
        was_training = self.trainer.model.training
        self.trainer.model.eval()
        with torch.inference_mode():

            label, predict = [], []
            train_loader = torch.utils.data.DataLoader(
                self.test_dataset,  # Replace with your dataset
                batch_size=32,         # Set batch size
                shuffle=False,         # No need to shuffle during inference
                collate_fn=collate_fn  # Use the same collate function
            )
            device = self.trainer.accelerator.device
            for batch in tqdm.tqdm(train_loader, disable=self.local_rank != 0):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                outputs = self.trainer.model(**inputs)
                label.append(batch['labels'].to(device))
                predict.append(outputs.logits.argmax(-1))
            label = torch.cat(label, dim=0)
            predict = torch.cat(predict, dim=0)
        if was_training:
            self.trainer.model.train()
        if self.world_size > 1 and dist.is_initialized():
            predict = gather_from_all_processes(predict)
            label = gather_from_all_processes(label)
        else:
            predict = predict.cpu().tolist()
            label = label.cpu().tolist()
        
        if self.local_rank == 0:
            correct_rate = sum(p==q for p,q in zip(predict, label)) / len(predict)
            if self.args.report_to == 'wandb':
                import wandb
                wandb.log({f'test/{self.args.task}-acc': correct_rate}, step=state.global_step)
            ans = {
                'task': self.args.task,
                'model': self.args.prj + '/' + str(state.global_step),
                'acc': round(correct_rate * 100, 2),
            }
            save_csv(ans, self.args.result)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', default="src.goat", choices=['src.goat', 'ft'])
    parser.add_argument('--model', default="openai/clip-vit-base-patch32", type=str)
    parser.add_argument('--result', default="../results", type=str)
    parser.add_argument('--prj', default="swift", type=str)
    parser.add_argument('--task', default="SVHN", choices=['Cars', 'DTD', 'GTSRB', 'RESISC45', 'SVHN'])
    parser.add_argument('--output', default="output", type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true', help="Enable gradient checkpointing.")
    parser.add_argument('--ep', default=1, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--bz', default=1, type=int)
    parser.add_argument('--gacc', default=1, type=int)
    parser.add_argument('--warmup', default=0, type=int)
    parser.add_argument('--sched', default="cosine", type=str)
    parser.add_argument('--swift_num_blocks', default=64, type=int, help='Number of SWIFT blocks (N).')
    parser.add_argument('--data_root', default='../data', type=str, help='Root directory for vision datasets.')
    parser.add_argument('--report_to', default='none', choices=['none', 'wandb'], help='Experiment tracker.')
    args = parser.parse_args()
    # assert args.lora in ["lora-pro", "rslora-pro"]
    return args      


def collate_fn(examples):
    # [b, 3, 224, 224]
    pixel_values = torch.stack([torch.tensor(example[0]) for example in examples])
    assert pixel_values.shape[1:] == (3,224,224)
    # [b, ]
    labels = torch.tensor([example[1] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def main():
    
    args = get_arguments()
    
    log.info(f"set seed to {args.seed}")
    transformers.set_seed(args.seed)
    set_seed(args.seed)
    
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    if rank == 0 and args.report_to == 'wandb':
        import wandb
        wandb.init(
            project=f'{args.model}-LoRA'.replace('/',''),  
            name=args.prj,
            config=args,
        )
    
    path = args.model
    image_processor = transformers.CLIPImageProcessor.from_pretrained(path)
    text_processor = AutoProcessor.from_pretrained(path)

    train_transform = Compose([
        Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        RandomResizedCrop(
            size=image_processor.size["shortest_edge"],
            scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC,
        ), 
        ToTensor(), 
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
    ])

    val_transform = Compose([
        Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        Resize(size=image_processor.size["shortest_edge"], interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(size=image_processor.size["shortest_edge"]),
        ToTensor(),
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    ])
    
    with TitledLog("load datasets and dataloaders", log_fn=log.info):
        from peta.tasks.cv import get_dataset
        try:
            dataset = get_dataset(
                args.task + 'Val',  # xxVal is the train set
                train_transform,
                location=args.data_root,
                batch_size=1,
            )
            test_dataset = get_dataset(
                args.task,  # xx is the test set
                val_transform,
                location=args.data_root,
                batch_size=1,
            ).test_dataset
        except (FileNotFoundError, RuntimeError) as exc:
            raise RuntimeError(
                f"Unable to load {args.task} from {args.data_root}. "
                "See image_classification/README.md for the required layout."
            ) from exc

    id2label = dataset.classnames
    label2id = {label: i for i, label in enumerate(id2label)}
    dataset = dataset.train_dataset

    # [goat setting: ]The classifier is obtained using prompts such as “a photo of a {class}” and kept frozen during fine-tuning
    model = transformers.CLIPForImageClassification.from_pretrained(
        path, 
        label2id=label2id,
        id2label=id2label, # set label number
        ignore_mismatched_sizes=True,  # Allow resizing the final classification layer
        # attn_implementation="flash_attention_2", 
        # torch_dtype=torch.bfloat16, 
        device_map={"": local_rank}
    )
    model.config.hidden_size = model.config.vision_config.hidden_size
    
    if 'ft' != args.lora:
        target_modules = 'all-linear'

        if 'src' in args.lora:

            import peta.src
            peta.src.register_swift_with_peft()
            lora_config = peta.src.GOATConfig(
                swift_num_blocks=args.swift_num_blocks,
                # r=lora_r,
                # use_rslora=True if "rs" in args.lora else False,
                # lora_alpha=lora_alpha,
                # lora_dropout=0.05,
                target_modules=target_modules,
                modules_to_save=['classifier'], 
                exclude_modules='classifier.*', # avoid add lora to untrained classifier head 
                # task_type="SEQ_CLS", cannot use peft seq cls task (only support NLP input_ids)
                bias="none",
                # num_experts=args.experts,
                # top_k=args.k,
                # init_type=init_type,
                # init_cof=float(init_cof),
            )
        
        if local_rank == 0:
            print('>>>', lora_config)

        model = peft.get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        if args.lora not in ["lora", "rslora"]: 
            for name, module in model.named_modules():
                if "lora_" in name:  
                    module.to(torch.float32)
    if local_rank == 0:
        unique_patterns = set()
        for n, p in model.named_parameters():
            # print(n)
            if p.requires_grad:
                if '.layer.' in n:
                    names = n.split('.layer.')
                    n = names[0].replace('base_model.', '') + '.' + '.'.join(names[1].split('.')[1:])
                elif '.layers.' in n:
                    names = n.split('.layers.')
                    n = names[0].replace('base_model.', '') + '.' + '.'.join(names[1].split('.')[1:])
                unique_patterns.add(n)
        print(unique_patterns)

    # trainer = CustomTrainer(
    trainer = Trainer(
        # scaling_factor=scaling_factor,  # fix
        model=model,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=image_processor,
        args=TrainingArguments(
            output_dir=args.output, 
            logging_dir="./transformers_logs",
            do_train=True,
            num_train_epochs=args.ep,
            per_device_train_batch_size=args.bz,
            gradient_accumulation_steps=args.gacc,
            optim="adamw_torch",
            logging_steps=1,
            bf16=True,
            learning_rate=args.lr,
            weight_decay=0, # No weight decay
            warmup_ratio=0.03, # warmup step override it 
            warmup_steps=args.warmup,
            lr_scheduler_type=args.sched,
            report_to="wandb" if rank == 0 and args.report_to == 'wandb' else "none",
            label_names=["labels"],  
            # ddp_find_unused_parameters=False if 'adamole' not in args.lora else True,
            ddp_find_unused_parameters=True, # clip for image classification donnot use the vision_model.post_layernorm, cause no grad
            evaluation_strategy='no',
            gradient_checkpointing=args.gradient_checkpointing,
            per_device_eval_batch_size=1,
            eval_steps=-1,
            save_strategy="no",
            save_steps=-1,
            save_total_limit=100,
            deepspeed="./config/deepspeed_zero2.json" if world_size > 1 and 'adamole' not in args.lora else None, 
         ),
        # data_collator=default_data_collator,
        data_collator=collate_fn,
        # cus_args = args,    #fix
    )
    trainer.add_callback(CustomCallback(trainer,test_dataset,args))
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    if rank == 0:
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        peta.src.save_swift_adapter(unwrapped_model, args.output)
        image_processor.save_pretrained(args.output)
        print(f'saved in {args.output}')
        if args.report_to == 'wandb':
            import wandb
            wandb.finish()

if __name__ == "__main__":
    main()
    
