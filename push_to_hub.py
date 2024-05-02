from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM
print('Start loading...')
local_path = '/root/llm_quantize/AutoGPTQ/Llama-2-7b-chat-4bit-gptq'
tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoGPTQForCausalLM.from_quantized(local_path, se_triton=True,
    use_safetensors=True,)
print('Loading done...')
print('Start Push Model...')
hf_repo = 'hoang1123/Llama-2-7b-chat-4bit-gptq'
tokenizer.push_to_hub(hf_repo)
model.push_to_hub(hf_repo)
print('Push Model done...')