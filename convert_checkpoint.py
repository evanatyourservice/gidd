import sys
import shutil
from pathlib import Path

from gidd.checkpoints import load_checkpoint
from gidd.models.configuration_dit import DITConfig
from gidd.models.modeling_dit import DIT


def main():
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    model, noise_schedule, tokenizer, config = load_checkpoint(input_path)

    hf_config = DITConfig(
        vocab_size=len(tokenizer),
        max_seq_len=config.model.max_seq_len,
        hidden_size=config.model.hidden_size,
        timestep_cond_dim=config.model.cond_dim,
        num_hidden_layers=config.model.n_blocks,
        num_attention_heads=config.model.n_heads,
        attention_dropout=config.model.dropout,
        p_uniform=config.model.p_uniform,
        t_eps=config.model.t_eps,
        auto_map={
            "AutoConfig": "configuration_dit.DITConfig",
            "AutoModelForMaskedLM": "modeling_dit.DIT"
        },
    )

    hf_model = DIT(hf_config)
    hf_model.load_state_dict(model.state_dict(), strict=False)

    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True)
    
    hf_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    shutil.copy("gidd/models/modeling_dit.py", output_path / "modeling_dit.py")
    shutil.copy("gidd/models/configuration_dit.py", output_path / "configuration_dit.py")


if __name__ == "__main__":
    main()
