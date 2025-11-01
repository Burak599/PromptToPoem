import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import evaluate
from torch.utils.tensorboard import SummaryWriter

# METRIC

metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
meteor_metric = evaluate.load("meteor")

# SETUP 
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATA 
data = load_dataset("checkai/instruction-poems")
split_dataset = data['train'].train_test_split(test_size=0.1, seed=42)
train_data_raw = split_dataset['train']
val_data_raw = split_dataset['test']

train_data_raw = train_data_raw.select(range(100))
val_data_raw = val_data_raw.select(range(100))


# Keep only input_text and target_text
def convert_to_input_target(example):
    return {"input_text": example["INSTRUCTION"], "target_text": example["RESPONSE"]}

train_data_raw = train_data_raw.map(convert_to_input_target, batched=True, remove_columns=train_data_raw.column_names)
val_data_raw = val_data_raw.map(convert_to_input_target, batched=True, remove_columns=val_data_raw.column_names)


# TOKENIZER & MODEL
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# LORA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)
model.to(device)

for name, param in model.named_parameters():
    param.requires_grad = "lora" in name.lower()

# PREPROCESS 
max_len = 128
def preprocess(example):
    inputs = tokenizer(example["input_text"], max_length=max_len, truncation=True, padding="max_length")
    targets = tokenizer(example["target_text"], max_length=max_len, truncation=True, padding="max_length")

    labels = np.array(targets["input_ids"])
    labels[labels == tokenizer.pad_token_id] = -100  # PAD token'larını maskele

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels.tolist()
    }

train_dataset = [preprocess(train_data_raw[i]) for i in range(len(train_data_raw))]
val_dataset = [preprocess(val_data_raw[i]) for i in range(len(val_data_raw))]

# PyTorch Dataset
class PoemDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long)
        }

train_dataset_pt = PoemDataset(train_dataset)
val_dataset_pt = PoemDataset(val_dataset)

batch_size = 4
train_loader = DataLoader(train_dataset_pt, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset_pt, batch_size=batch_size, num_workers=0)

# OPTIMIZER & SCHEDULER 
grad_clip = 1.0
num_epochs = 4
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-6)
total_steps = num_epochs * len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=2e-5,
    total_steps=total_steps,
    pct_start=0.3,
    anneal_strategy="cos",
    div_factor=25.0,
    final_div_factor=1e4
)

# MANUAL TRAINING LOOP 
log_every = 100
val_every = 500
writer = SummaryWriter("./runs/poem_training")
global_step = 0

for epoch in range(num_epochs):
    model.train()
    total_loss, total_correct, total_count = 0, 0, 0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(outputs.logits, dim=-1)
        mask = labels != -100
        correct = (preds[mask] == labels[mask]).sum().item()
        total = mask.sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_count += total

        global_step += 1
        train_acc = 100 * total_correct / total_count if total_count > 0 else 0
        writer.add_scalar("Loss/train", loss.item(), global_step)
        writer.add_scalar("Accuracy/train", train_acc, global_step)

        # Train progress bar 
        progress = 100 * ((step % log_every) + 1) / log_every
        bar_length = 20
        filled_length = int(bar_length * progress // 100)
        bar = "#" * filled_length + "-" * (bar_length - filled_length)
        print(f"\rTrain Progress: [{bar}] {progress:.0f}%", end="")

        # Validation 
        if (step + 1) % val_every == 0:
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            all_preds = []
            all_labels = []

            for val_batch in val_loader:
                input_ids_val = val_batch["input_ids"].to(device)
                attention_mask_val = val_batch["attention_mask"].to(device)
                labels_val = val_batch["labels"].to(device)

                outputs_val = model(input_ids=input_ids_val, attention_mask=attention_mask_val, labels=labels_val)
                val_loss += outputs_val.loss.item()

                preds_val = torch.argmax(outputs_val.logits, dim=-1)
                mask_val = labels_val != -100
                val_correct += (preds_val[mask_val] == labels_val[mask_val]).sum().item()
                val_total += mask_val.sum().item()

                decoded_preds = tokenizer.batch_decode(preds_val, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels_val, skip_special_tokens=True)
                all_preds.extend(decoded_preds)
                all_labels.extend(decoded_labels)

            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            val_loss_avg = val_loss / len(val_loader)

            metric_score = metric.compute(predictions=all_preds, references=all_labels)
            rouge_l = metric_score["rougeL"].mid.fmeasure
            bleu_preds = [pred.split() for pred in all_preds]
            bleu_refs = [[ref.split()] for ref in all_labels]
            bleu_score = bleu_metric.compute(predictions=bleu_preds, references=bleu_refs)["bleu"]
            meteor_score = meteor_metric.compute(predictions=all_preds, references=all_labels)["meteor"]

            writer.add_scalar("RougeL/val", rouge_l, global_step)
            writer.add_scalar("BLEU/val", bleu_score, global_step)
            writer.add_scalar("METEOR/val", meteor_score, global_step)

            train_acc = 100 * total_correct / total_count if total_count > 0 else 0
            print(f"\n[Epoch {epoch+1}] Step {step+1}/{len(train_loader)} | "
                  f"Train Loss: {total_loss/(5 * log_every):.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%\n")

            # Reset train counters
            total_loss, total_correct, total_count = 0, 0, 0
            model.train()

# GENERATE POEM 
prompt = "Write me a poem about Religion"
model.eval()
with torch.no_grad():
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=128,       
        do_sample = True, 
        temperature=0.82,   
        top_p=0.82,
        repetition_penalty=2.5,   
        no_repeat_ngram_size=3,  
        early_stopping=True
    )
    print("\nGenerated poem:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))

# SAVE MODEL
inp = input("Type 'ok' to save the model: ")
if inp.lower() == "ok":
    model.save_pretrained("./Model_Poem2")
    tokenizer.save_pretrained("./Model_Poem2")