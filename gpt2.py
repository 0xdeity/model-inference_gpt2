from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
text = "Hi what is your name and what do you do here?"

encoded_input = tokenizer.encode(text, return_tensors='pt')
output = model.generate(encoded_input, max_length=200, do_sample=True, no_repeat_ngram_size=2, early_stopping=True)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print(response)
