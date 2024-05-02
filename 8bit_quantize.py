import torch
import time

def absmax_quantize(X):
    # Calculate scale
    scale = 127 / torch.max(torch.abs(X))

    # Quantize
    X_quant = (scale * X).round()

    # Dequantize
    X_dequant = X_quant / scale

    return X_quant.to(torch.int8), X_dequant

def zeropoint_quantize(X):
    # Calculate value range (denominator)
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range

    # Calculate scale
    scale = 255 / x_range

    # Shift by zero-point
    zeropoint = (-scale * torch.min(X) - 128).round()

    # Scale and round the inputs
    X_quant = torch.clip((X * scale + zeropoint).round(), -128, 127)

    # Dequantize
    X_dequant = (X_quant - zeropoint) / scale

    return X_quant.to(torch.int8), X_dequant

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    global device, model, tokenizer,model_id

    # Set device to CPU for now
    device = 'cpu'

    # Load model and tokenizer
    model_id = 'NousResearch/Llama-2-7b-chat-hf'
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Print model size
    print(f"Model size: {model.get_memory_footprint():,} bytes")
    # time.sleep(20)

def naive_8bit_quantize():
    from copy import deepcopy
    print('start quantize')
    # Store original weights
    weights = [param.data.clone() for param in model.parameters()]

    # Create model to quantize
    model_abs = deepcopy(model)

    # Quantize all model weights
    # weights_abs = []
    print('quanztize abs')
    for param in model_abs.parameters():
        _, dequantized = absmax_quantize(param.data)
        param.data = dequantized
        # weights_abs.append(dequantized)

    # Create model to quantize
    print('quantize zp')
    model_zp = deepcopy(model)

    # Quantize all model weights
    # weights_zp = []
    for param in model_zp.parameters():
        _, dequantized = zeropoint_quantize(param.data)
        param.data = dequantized
        # weights_zp.append(dequantized)
    
    return model_abs, model_zp
    
def calculate_perplexity(model, text):
    # Encode the text
    encodings = tokenizer(text, return_tensors='pt').to(device)

    # Define input_ids and target_ids
    input_ids = encodings.input_ids
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

    # Loss calculation
    neg_log_likelihood = outputs.loss

    # Perplexity calculation
    ppl = torch.exp(neg_log_likelihood)

    return ppl

def llm_int8_quantize():
    model_int8 = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map='auto',
                                             load_in_8bit=True,
                                             )
    print(f"Model size: {model_int8.get_memory_footprint():,} bytes")

if __name__ == "__main__":
    load_model()
    model_abs, model_zp = naive_8bit_quantize()

    model_abs.save_pretrained('Llama-2-7b-chat-hf-8bit-abs')
    tokenizer.save_pretrained('Llama-2-7b-chat-hf-8bit-abs')
    model_zp.save_pretrained('Llama-2-7b-chat-hf-8bit-zp')
    tokenizer.save_pretrained('Llama-2-7b-chat-hf-8bit-zp')

    # print(model)
    # print(model.model.layers[0].self_attn.q_proj.weight.data)
    # exit()
    # for name, param in model.named_parameters():
    #     print(name)
    # Extract weights of the first layer
    # weights = model.model.layers[0].self_attn.q_proj.weight.data
    # print("Original weights:")
    # print(weights)

    # # Quantize layer using absmax quantization
    # weights_abs_quant, _ = absmax_quantize(weights)
    # print("\nAbsmax quantized weights:")
    # print(weights_abs_quant)

    # # Quantize layer using absmax quantization
    # weights_zp_quant, _ = zeropoint_quantize(weights)
    # print("\nZero-point quantized weights:")
    # print(weights_zp_quant)
