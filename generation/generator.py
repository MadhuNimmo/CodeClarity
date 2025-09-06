from typing import Dict
import torch

def generate_summary(code: str,
                     language_name: str,
                     prompt_template: str,
                     model, tokenizer, device: str,
                     apply_chat, gen_kwargs: Dict) -> str:
    """
    Notebook-equivalent text generation:
    - build user prompt via chat template
    - generate with nucleus-style settings
    - return only generated continuation (tail after input)
    """
    user_prompt = prompt_template.format(code=code, language_name=language_name)
    messages = [{"role": "user", "content": user_prompt}]
    prompt = apply_chat(messages)

    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.1,
            do_sample=True,
            top_p=0.1,
            **gen_kwargs
        )

    # Keep only what was generated beyond the input prompt
    tail_ids = outputs[:, inputs["input_ids"].shape[-1]:]
    text = tokenizer.batch_decode(tail_ids, skip_special_tokens=True)[0].strip()
    return text
