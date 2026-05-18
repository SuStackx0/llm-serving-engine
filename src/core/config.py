from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hidden_size: int = 2048
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    intermediate_size: int = 5632
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def gqa_ratio(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf_config(cls, hf_cfg: dict, model_id: str = "") -> "ModelConfig":
        return cls(
            model_id=model_id,
            hidden_size=hf_cfg.get("hidden_size", 2048),
            num_hidden_layers=hf_cfg.get("num_hidden_layers", 22),
            num_attention_heads=hf_cfg.get("num_attention_heads", 32),
            num_key_value_heads=hf_cfg.get("num_key_value_heads", 4),
            intermediate_size=hf_cfg.get("intermediate_size", 5632),
            vocab_size=hf_cfg.get("vocab_size", 32000),
            max_position_embeddings=hf_cfg.get("max_position_embeddings", 2048),
            rms_norm_eps=hf_cfg.get("rms_norm_eps", 1e-5),
            rope_theta=hf_cfg.get("rope_theta", 10000.0),
        )


@dataclass
class EngineConfig:
    # KV cache block config
    block_size: int = 16            # tokens per physical block
    num_blocks: int = 256           # total physical blocks

    # Scheduler limits
    max_running_requests: int = 8
    max_waiting_requests: int = 256

    # Preemption
    preemption_mode: str = "requeue"   # "requeue" or "drop"

    # Device
    device: str = "auto"    # "auto" | "mps" | "cpu" | "cuda"
    dtype: str = "auto"     # "auto" | "float32" | "float16"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def resolve_dtype(self, device: str):
        import torch
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "float32":
            return torch.float32
        # auto: float32 on MPS/CPU (stability), float16 on CUDA
        return torch.float16 if device == "cuda" else torch.float32


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
