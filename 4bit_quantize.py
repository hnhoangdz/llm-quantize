from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
quant_config = BitsAndBytesConfig(
   load_in_4bit=True, # 4bit load precision
   bnb_4bit_quant_type="nf4", # nf4 data type (better results)
   bnb_4bit_use_double_quant=True, # 
   bnb_4bit_compute_dtype=torch.bfloat16
)

#uncomment for 8bit precision
"""quant_config = BitsAndBytesConfig(
    load_in_8bit=True
)"""

#uncomment for 4bit precision
"""quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
) """

model_id = 'NousResearch/Llama-2-7b-chat-hf'
model = AutoModelForCausalLM.from_pretrained(model_id,
device_map = "auto", quantization_config = quant_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = [
    {"role": "user", "content": "What is Natural Language Processing?"}
]
encodeds = tokenizer.apply_chat_template(prompt, return_tensors="pt")

model_inputs = encodeds.to("cuda")

generated_ids = model.generate(model_inputs, max_new_tokens=200, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])