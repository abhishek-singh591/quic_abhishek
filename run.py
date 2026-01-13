from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-1B"
model1 = QEFFAutoModelForCausalLM.from_pretrained(model_name)
model1.compile(num_devices=1, num_cores=16, ctx_len=8192)
hash_0_1 = model1.export_hash
inputs = "Help me with this"
tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_00 = model1.generate(prompts=["Help me with this"], tokenizer=tokenizer)
