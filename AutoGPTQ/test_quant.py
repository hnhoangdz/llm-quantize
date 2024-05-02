from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from transformers import pipeline
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = 'hoang1123/Llama-2-7b-chat-4bit-gptq'
model = AutoGPTQForCausalLM.from_quantized(
    model_id,
    device=device,
    use_triton=True,
    use_safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
result = generator("I have a dream", do_sample=True, max_length=500)[0]['generated_text']
print(result)