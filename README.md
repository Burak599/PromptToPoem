# T5 LoRA Poem Generator

This project provides a complete framework for fine-tuning **T5 (Text-to-Text Transfer Transformer)** models with **LoRA (Low-Rank Adaptation)** to generate poems from natural language instructions. The goal is to train a model that can understand a textual prompt and produce creative, coherent, and stylistically appropriate poetry.  

Unlike traditional fine-tuning, LoRA allows us to adapt large language models efficiently by updating only a small number of low-rank parameters. This approach reduces memory usage and speeds up training while still achieving high-quality generation.  

## Features

- **Model**: Uses `t5-base` as the backbone transformer model.  
- **LoRA Fine-Tuning**: Efficient parameter adaptation on attention layers.  
- **Dataset**: `checkai/instruction-poems` with prompts (`INSTRUCTION`) and target poems (`RESPONSE`).  
- **Evaluation Metrics**: Supports ROUGE-L, BLEU, and METEOR for quantitative evaluation.  
- **Manual Training Loop**: PyTorch-based training with gradient clipping and learning rate scheduling.  
- **Text Generation**: Supports stochastic text generation with temperature, top-p sampling, repetition penalties, and no-repeat n-gram constraints.  
- **TensorBoard Logging**: Tracks training loss, accuracy, and evaluation metrics.  

## Requirements

- Python 3.10+  
- PyTorch 2.x  
- Transformers  
- Datasets  
- PEFT (for LoRA)  
- Evaluate (for ROUGE, BLEU, METEOR)  
- TensorBoard (optional, for monitoring)  

## Installation

```bash
pip install torch transformers datasets peft evaluate tensorboard
tensorboard --logdir=runs
