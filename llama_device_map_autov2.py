import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer
import pickle
import pynvml

class LlamaConfig:
    '''
    Define the model parameters : dimension, numberof layers, etc...
    Must be retrieved from the config.json file
    '''
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        intermediate_size=11008,
        max_position_embeddings=2048,
        pad_token_id=0,
        rms_norm_eps=1e-05,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        num_key_value_heads=16,
        rope_theta=10000,
        rope_scaling=None,  # Add this line
    ):
        self.rope_scaling = rope_scaling  # Add this line
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta


#Doesn't center the datas (substraction by the mean)
class LlamaRMSNorm(nn.Module):
    '''
    Llama normalization (root mean square), doesn't center the datas (substraction by the mean)
    '''
    def __init__(self, hidden_size, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # compute mean of square on last dimension (variance)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # normalisation
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    # Divide the vectors in two parts
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # 90° rotation in complex space
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _compute_default_rope_parameters(config: LlamaConfig, device=None, seq_len=None):
    base = config.rope_theta
    head_dim = config.hidden_size // config.num_attention_heads
    dim = head_dim
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq, 1.0


def _compute_llama3_parameters(config: LlamaConfig, device=None, seq_len=None):
    '''
    Utilty function to compute Rope parameters for llama
    Called by LlamaRotaryEmbedding init
    '''
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device)
    scaling = config.rope_scaling
    factor = scaling["factor"]
    low_freq_factor = scaling["low_freq_factor"]
    high_freq_factor = scaling["high_freq_factor"]
    original_max_pos = scaling["original_max_position_embeddings"]

    low_freq_wavelen = original_max_pos / low_freq_factor
    high_freq_wavelen = original_max_pos / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (original_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth_factor) * (inv_freq / factor) + smooth_factor * inv_freq
    is_medium = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
    inv_freq_llama = torch.where(is_medium, smoothed, inv_freq_llama)

    return inv_freq_llama, attention_factor


ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "llama3": _compute_llama3_parameters,
}

class LlamaRotaryEmbedding(nn.Module):
    '''
    Handle llama rotary embedding
    '''
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", "default")
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LlamaMLP(nn.Module):
    '''
    MLP, module feedforward of the llama model
    Present in each layer of the model
    Down(Silu(Gate(x) * Proj(x)))
    '''
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    '''
    Because Llama has a Grouped Query Attention architecture, num heads > num kv heads
    Repeat KV matrix to make theme fit with number of queries head
    '''
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    #add a dimension to repeatf
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    #merge the repeated heads
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)

class LlamaAttention(nn.Module):
    '''
    Attention module
    '''
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.q_debug = None
        self.k_debug = None

    def forward(self, x, cos, sin):
        batch_size, seq_len, _ = x.shape
        #compute Q, K and V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)


        # Apply rotary embeddings with correct unsqueeze
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # Store outputs in attributes for later debugging


        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        rep_factor = self.num_heads // self.num_kv_heads  # e.g., 64/16 = 4
        k = repeat_kv(k, rep_factor)  # becomes [batch, 64, seq_length, head_dim]
        v = repeat_kv(v, rep_factor)
        # Q.K^T , Transpose K on the last two dimensions and then normalisation ( * 1/sqrt(head_dim))
        attn_weights = (q @ k.transpose(-2, -1)) * self.scaling
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        return self.o_proj(attn_output)


class LlamaDecoderLayer(nn.Module):
    """
    A layer of the model which contains :
    - attention
    - MLP
    - RMS normalizations
    """
    def __init__(self, config):
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, cos, sin):
        # attention(norm(x)) + x
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin)
        x = residual + x

        # mlp(norm(x)) + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Get available devices
        self.devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        if not self.devices:
            self.devices = [torch.device('cpu')]

        # Initialize components on appropriate devices
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size).to(self.devices[0])

        # Distribute layers across available devices
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            device_idx = i % len(self.devices)
            layer = LlamaDecoderLayer(config).to(self.devices[device_idx])
            self.layers.append(layer)

        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps).to(self.devices[0])
        self.rotary_emb = LlamaRotaryEmbedding(config).to(self.devices[0])

    def forward(self, input_ids):
        # Ensure input is on first device
        input_device = self.devices[0]
        input_ids = input_ids.to(input_device)

        batch_size, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        # Generate position_ids on first device
        position_ids = torch.arange(seq_len, device=input_device).unsqueeze(0).expand(batch_size, -1)
        cos, sin = self.rotary_emb(x, position_ids)

        # Process through layers with device transfers
        for i, layer in enumerate(self.layers):
            layer_device = self.devices[i % len(self.devices)]

            # Transfer tensors to layer device
            x = x.to(layer_device)
            cos_layer = cos.to(layer_device)
            sin_layer = sin.to(layer_device)

            x = layer(x, cos_layer, sin_layer)

        # Final norm on first device
        x = x.to(input_device)
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(self.model.devices[0])
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states)

# Example usage
if __name__ == "__main__":
    #from config.json
    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=2048,
        num_hidden_layers=16,
        num_attention_heads=32,
        intermediate_size=8192,
        max_position_embeddings=131072,
        num_key_value_heads=8,
        pad_token_id=0,
        rms_norm_eps=1e-05,
        rope_theta=500000,
        rope_scaling= {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
    )

    print("VRAM USAGE BEFORE LOADING")
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        print(f"GPU {i} ({name}):")
        print(f"  Total memory: {mem_info.total / (1024 ** 2):.2f} MB")
        print(f"  Used memory : {mem_info.used / (1024 ** 2):.2f} MB")
        print(f"  Free memory : {mem_info.free / (1024 ** 2):.2f} MB")
        print()

    pynvml.nvmlShutdown()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model without manual device placement
    model = LlamaForCausalLM(config)

    # Load state dict (automatically handles device placement because llamaforcausallm class is shared across devices)
    state_dict = load_file("Llama-3.2-1B-Instruct/model.safetensors")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("Llama-3.2-1B-Instruct")

    print("VRAM USAGE AFTER LOADING")
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        print(f"GPU {i} ({name}):")
        print(f"  Total memory: {mem_info.total / (1024 ** 2):.2f} MB")
        print(f"  Used memory : {mem_info.used / (1024 ** 2):.2f} MB")
        print(f"  Free memory : {mem_info.free / (1024 ** 2):.2f} MB")
        print()

    pynvml.nvmlShutdown()
    # Inputs automatically go to first device
    prompt = "I'm from Paris and my favorite neighbourhood is "
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.model.devices[0])

    # Generation loop remains the same
    output = ""
    for _ in range(10):
        with torch.no_grad():
            logits = model(input_ids)
            last_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(last_token_logits, dim=-1)
            input_ids = torch.cat((input_ids, next_token_id.unsqueeze(1).to(model.model.devices[0])), dim=1)
            output += tokenizer.decode(next_token_id, skip_special_tokens=True)

    print("Generated:", output)
