import gc
import torch
import time
from quantized_llama import modeling_llama
import transformers

model_configs = [
    "meta-llama/Llama-2-7b-hf",
    #"meta-llama/Llama-2-13b-hf", 
    # "meta-llama/Llama-2-70b-hf", 
]

benchmark_dtypes = ["int4", torch.float16]
num_warmup_steps = 3
num_bench_steps = 10

def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def module_benchmark(module):
    # warmup
    for i in range(num_warmup_steps):
        out = module()
    torch.cuda.synchronize()
    
    _cleanup()
    torch.cuda.reset_max_memory_allocated()
    start_time = time.perf_counter()
    
    for i in range(num_bench_steps):
        out = module()
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()

    end_time = time.perf_counter()

    return (end_time - start_time) * 1000 / num_bench_steps, peak_memory


class TimedModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.total_time = 0.0
        self.call_count = 0

    def forward(self, x):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.module(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        self.total_time += (end - start) * 1000
        self.call_count += 1
        return out


# main benchmarking function
# -> the method is to flatten the hierarchy of functions (like the one in the main benchmark function)
# -> after flattening, insert timing function around the hadamard steps in the model
# -> (according the QuaRot's E2E code, online hadamard transformation is only applied on attention o-project and ffn down-proj)
# -> after inserting timer, run the inference as usual
# -> collect data!

def layer_flatten(batch_size, prefill_len, decode_step):
    # - - - CONFIG - - - 
    config_name = model_configs[0]
    config = transformers.AutoConfig.from_pretrained(
        config_name,
        attn_implementation="flash_attention_2"
    )
    
    # ---- PARAMS ----
    num_hadamard_calls_per_layer = 2
    num_layers = 32
    
    # ---- GENERATE MODEL ---- 
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights(): 
        model = modeling_llama.QuarotLlamaForCausalLM(config=config)
    torch.set_default_dtype(dtype_old)
    
    # ---- MODEL SETUP ----
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) 
    print(next(model.parameters()).device)
    
    for name, param in model.named_parameters():
        if param.device != device:
            print(f"Parameter {name} is on {param.device}, expected {device}")
    
    
    # add timing modules around the layers
    for i, layer in enumerate(model.model.layers):
        
        # mlp (ffn)
        layer.mlp.quantizer = TimedModule(layer.mlp.quantizer)
        layer.mlp.up_proj  = TimedModule(layer.mlp.up_proj )
        layer.mlp.gate_proj = TimedModule(layer.mlp.gate_proj)
        hadamard = layer.mlp.down_proj[0]
        layer.mlp.down_proj[0] = TimedModule(hadamard)
        
        # attention 
        layer.mlp.quantizer = TimedModule(layer.mlp.quantizer)
        layer.self_attn.q_proj = TimedModule(layer.self_attn.q_proj)
        layer.self_attn.k_proj = TimedModule(layer.self_attn.k_proj)
        layer.self_attn.v_proj = TimedModule(layer.self_attn.v_proj)
        layer.self_attn.o_proj[0] = TimedModule(layer.self_attn.o_proj[0])
        layer.self_attn.o_proj[1] = TimedModule(layer.self_attn.o_proj[1])
        hadamard = layer.self_attn.o_proj_hadamard
        layer.self_attn.o_proj_hadamard = TimedModule(hadamard)

    
    
    # ---- INFERENCE (E2E TIME) ----
    device = model.device
    test_input = torch.randint(100, 200, (batch_size, prefill_len), dtype=torch.int32, device=device)
    next_input = torch.tensor([[100] for _ in range (batch_size)], dtype=torch.int32, device=device)
    
    def _prefill_and_decode_for_multiple_steps():
        model._expected_max_length = prefill_len + decode_step
        out = model(test_input)
        for _ in range(decode_step):
            model(next_input, past_key_values=out.past_key_values)  
    time_e2e, _ = module_benchmark(_prefill_and_decode_for_multiple_steps)
    
    # ---- CALCULATE RESULTS ----
    mlp_total_time = 0
    mlp_call_count = 0

    attn_total_time = 0
    attn_call_count = 0

    online_hadamard_time = 0
    hadamard_call_count  = 0
    
    for layer in model.model.layers:
        # mlp components
        for proj in [layer.mlp.quantizer, layer.mlp.up_proj, layer.mlp.gate_proj, layer.mlp.down_proj[0]]:
            mlp_total_time += proj.total_time
            mlp_call_count += proj.call_count

        # self-attention components
        for proj in [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj[0],
            layer.self_attn.o_proj[1],
        ]:
            attn_total_time += proj.total_time
            attn_call_count += proj.call_count

        # hadamard
        timed_mlp_hadamard = layer.mlp.down_proj[0]
        online_hadamard_time += timed_mlp_hadamard.total_time
        hadamard_call_count  += timed_mlp_hadamard.call_count

        timed_attn_hadamard = layer.self_attn.o_proj_hadamard
        online_hadamard_time += timed_attn_hadamard.total_time
        hadamard_call_count  += timed_attn_hadamard.call_count

    # ---- PRINT RESULTS ----
    print("=" * 60)
    print(f"[Batch size: {batch_size}] | [Prefill: {prefill_len}] | [Decode: {decode_step}]")
    print(f"E2E Inference Time (ms): {time_e2e:.2f}")

    if hadamard_call_count > 0:
        avg_hadamard_time = online_hadamard_time / hadamard_call_count
        print(f"Avg Hadamard Time per Call (ms): {avg_hadamard_time:.6f}")
        estimated_total_hadamard_time = avg_hadamard_time * num_hadamard_calls_per_layer * num_layers * (1 + decode_step)
        print(f"Estimated Total Hadamard Time (ms): {estimated_total_hadamard_time:.2f}")
    else:
        print("No Hadamard calls detected.")

    if mlp_call_count > 0:
        print(f"Avg MLP Component Time per Call (ms): {mlp_total_time / mlp_call_count:.6f}")
    if attn_call_count > 0:
        print(f"Avg Attention Component Time per Call (ms): {attn_total_time / attn_call_count:.6f}")
    print("=" * 60)
    
    # CLEAN UP
    del model
    _cleanup() 
    
    
if __name__ == '__main__':
    batch_sizes = [1]
    prefill_len = 1024
    decode_lens = [512]

    for batch_size in batch_sizes:
        for decode_len in decode_lens:
            layer_flatten(batch_size, prefill_len, decode_len)