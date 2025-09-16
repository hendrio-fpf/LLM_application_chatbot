from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.eos_token_id

print("🤖 BlenderBot iniciado! Digite 'exit' para sair.\n")

while True:
    user_input = input("Você: ").strip()
    if user_input.lower() in ["exit", "quit", "sair"]:
        break

    # Tokenizar (sem forçar </s> no fim)
    inputs = tokenizer(
        user_input,
        return_tensors="pt",
        add_special_tokens=True
    )

    # Remover eos se já vier no final
    if inputs["input_ids"][0, -1].item() == tokenizer.eos_token_id:
        inputs["input_ids"] = inputs["input_ids"][:, :-1]
        inputs["attention_mask"] = inputs["attention_mask"][:, :-1]

    print(f"\n[DEBUG] Tokenized input_ids: {inputs['input_ids']}")

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        pad_token_id=model.config.pad_token_id
    )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(f"Bot: {reply}\n")
