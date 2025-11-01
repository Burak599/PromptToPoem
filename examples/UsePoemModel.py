from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "./Model_Poem2"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

print(f"Model loaded on {device}. Type 'exit' to quit.\n")

while True:
    prompt = input("Enter a prompt: ")
    if prompt.lower() == "exit":
        print("Exiting...")
        break

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            do_sample=True,
            temperature=0.82,
            top_p=0.82,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    poem = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- Intro kısmını ayır ---
    parts = poem.split(':', 1)
    if len(parts) == 2:
        intro, poem = parts
        print(f"\n{intro.strip()}:\n")
    else:
        intro = None

    # --- Şiir biçimlendirme ---
    for punc in ['. ', '; ', ', ', '! ', '? ']:
        poem = poem.replace(punc, punc.strip() + '\n')

    formatted_lines = []
    for line in poem.split('\n'):
        words = line.split()
        for i in range(0, len(words), 12):
            formatted_lines.append(' '.join(words[i:i+12]))
    poem = '\n'.join([line for line in formatted_lines if line.strip()])

    print("\nGenerated poem:\n")
    print(poem)
    print("\n" + "="*60 + "\n")
