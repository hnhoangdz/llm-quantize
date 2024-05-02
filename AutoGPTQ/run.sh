python examples/quantization/quant_with_alpaca.py \
--pretrained_model_dir NousResearch/Llama-2-7b-chat-hf \
--quantized_model_dir Llama-2-7b-chat-4bit-gptq \
--bit 4 \
--group_size 128 \
--num_samples 1000 \
--save_and_reload \
--use_triton \
--quant_batch_size 4 \
--trust_remote_code \
