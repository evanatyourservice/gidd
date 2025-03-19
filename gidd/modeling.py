from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import gidd.models.dit as dit

try:
    import flash_attn
    has_flash_attn = True
except ImportError:
    has_flash_attn = False


def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    tokenizer.model_max_length = config.model.max_seq_len
    return tokenizer


def get_model(config, tokenizer, device=None, dtype=None):
    if config.model.type == "diffusion":
        model = dit.DIT(config, len(tokenizer))
    elif config.model.type == "autoregressive":
        cfg = LlamaConfig(
            vocab_size=len(tokenizer),
            num_hidden_layers=config.model.n_blocks,
            hidden_size=config.model.hidden_size,
            intermediate_size=4*config.model.hidden_size,
            num_attention_heads=config.model.n_heads,
            max_position_embeddings=config.model.max_seq_len,
            attn_implementation="flash_attention_2" if has_flash_attn else "sdpa",
            torch_dtype=dtype,
        )
        model = LlamaForCausalLM(cfg)
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    if device is not None:
        model = model.to(device, dtype=dtype)
    elif dtype is not None:
        model = model.to(dtype=dtype)

    return model
