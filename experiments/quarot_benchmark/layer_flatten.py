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

def layer_flatten():
    # CONFIG
    config_name = model_configs[0]
    config = transformers.AutoConfig.from_pretrained(
        config_name,
        attn_implementation="flash_attention_2"
    )
    
    # PARAMS
    batch_size = 2
    prefill_len = 1024
    decode_step = 512
    
    # GENERATE MODEL
    dtype_old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    with transformers.modeling_utils.no_init_weights(): 
        model = modeling_llama.QuarotLlamaForCausalLM(config=config)
    torch.set_default_dtype(dtype_old)
    
    # MODEL SETUP
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) 
    print(next(model.parameters()).device)
    
    for name, param in model.named_parameters():
        if param.device != device:
            print(f"Parameter {name} is on {param.device}, expected {device}")
    
    
    # ADD TIME MODULE AROUND THE HADAMARD LAYERS
    for i, layer in enumerate(model.model.layers):
        # mlp (ffn)
        hadamard = layer.mlp.down_proj[0]
        layer.mlp.down_proj[0] = TimedModule(hadamard)
        # attention 
        hadamard = layer.self_attn.o_proj_hadamard
        layer.self_attn.o_proj_hadamard = TimedModule(hadamard)
        
        # debug prints to make sure they are not repeated
        print(f"[Layer {i}] MLP TimedModule ID:", id(layer.mlp.down_proj[0]))
        print(f"[Layer {i}] Attn TimedModule ID:", id(layer.self_attn.o_proj_hadamard))
    
    
    # INFERENCE (E2E TIME)
    device = model.device
    test_input = torch.randint(100, 200, (batch_size, prefill_len), dtype=torch.int32, device=device)
    next_input = torch.tensor([[100] for _ in range (batch_size)], dtype=torch.int32, device=device)
    
    def _prefill_and_decode_for_multiple_steps():
        model._expected_max_length = prefill_len + decode_step
        out = model(test_input)
        for _ in range(decode_step):
            model(next_input, past_key_values=out.past_key_values)
            
    time_e2e, _ = module_benchmark(_prefill_and_decode_for_multiple_steps)
    
    # CALCULATE TOTAL HADAMARD TIME
    online_hadamard_time = 0
    hadamard_call_count  = 0
    
    for i, layer in enumerate(model.model.layers):
        print(f"Layer {i} - MLP calls: {layer.mlp.down_proj[0].call_count}, "
            f"Attn calls: {layer.self_attn.o_proj_hadamard.call_count}")


    for layer in model.model.layers:
        # MLP(ffn) hadamard
        timed_mlp = layer.mlp.down_proj[0]
        online_hadamard_time += timed_mlp.total_time
        hadamard_call_count  += timed_mlp.call_count

        # attention hadamard
        timed_attn = layer.self_attn.o_proj_hadamard
        online_hadamard_time += timed_attn.total_time
        hadamard_call_count  += timed_attn.call_count
    
    avg_time_per_call = online_hadamard_time / hadamard_call_count
    print("avg hadamard time per call (ms):", avg_time_per_call)


    print("total time (ms): ", time_e2e)
    
    estimated_total_hadamard_time = avg_time_per_call * 64 * (1 + decode_step) * batch_size
    print("online hadamard time estimate (ms): ", estimated_total_hadamard_time)
    
    # CLEAN UP
    del model
    _cleanup() 
    
if __name__ == '__main__':
    layer_flatten()  