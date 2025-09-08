from typing import Tuple, Callable, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def _device_dtype():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    return device, dtype

def _generic_loader(model_id: str, trust_remote_code: bool = False):
    device, dtype = _device_dtype()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=dtype, trust_remote_code=trust_remote_code
    ).to(device)

    def apply_chat(messages):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # pad/eos where available; safe defaults
    gen_kwargs = {}
    if getattr(tok, "eos_token_id", None) is not None:
        gen_kwargs["pad_token_id"] = tok.eos_token_id
        gen_kwargs["eos_token_id"] = tok.eos_token_id
    return mdl, tok, device, apply_chat, gen_kwargs

def load_codegemma(model_id="google/codegemma-7b-it"):
    return _generic_loader(model_id)

def load_gemma(model_id="google/gemma-2-9b-it"):
    return _generic_loader(model_id)

def load_qwen(model_id="Qwen/Qwen2.5-Coder-7B-Instruct"):
    return _generic_loader(model_id, trust_remote_code=True)

def load_deepseek(model_id="deepseek-ai/deepseek-coder-6.7b-instruct"):
    return _generic_loader(model_id, trust_remote_code=True)

REGISTRY: Dict[str, Callable[..., Tuple[Any, Any, str, Callable, Dict]]] = {
    "codegemma": load_codegemma,
    "gemma":     load_gemma,
    "qwen":      load_qwen,
    "deepseek":  load_deepseek,
}
