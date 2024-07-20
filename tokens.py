from transformers import AutoTokenizer


def token_size(input_text):
    # Load the pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token="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")

    tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=0,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
        padding='max_length'
    )['input_ids'].flatten().tolist()

    return len(tokens)+40


# prompt="You are a translator bot, translate Hi! My name is to French Only give the translated text."
# print(token_size(prompt))
