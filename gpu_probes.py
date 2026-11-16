"""Optional GPU runner for the two named checkpoints; demo.py needs no GPU.

Run in separate processes to release GPU memory between checkpoints. Pin each
revision explicitly. Both models receive the same serialized user prompt and
empty thinking prefix. These are text-only probes, not an omics evaluation.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=['ingenix/Thalamus-Reasoning-9B', 'Qwen/Qwen3.5-9B'])
    p.add_argument('--revision', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a fresh path to preserve the raw run.')
    from huggingface_hub import snapshot_download
    import torch
    from vllm import LLM, SamplingParams
    checkpoint = snapshot_download(args.model, revision=args.revision,
        allow_patterns=['*.json', '*.safetensors', '*.py', '*.jinja'])
    thalamus = args.model.startswith('ingenix/')
    if thalamus:
        from thalamus_reasoning.vllm_plugin import register_thalamus_reasoning
        register_thalamus_reasoning()
    settings = dict(temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
                    max_tokens=1024, seed=0, skip_special_tokens=True)
    engine = dict(model=checkpoint, trust_remote_code=True, dtype='bfloat16',
                  max_model_len=4096, gpu_memory_utilization=0.85,
                  enforce_eager=True, seed=0, disable_log_stats=True,
                  limit_mm_per_prompt={'image': 1 if thalamus else 0})
    if thalamus:
        engine['enable_mm_embeds'] = True
    llm = LLM(**engine)
    prompts = json.loads(Path(__file__).with_name('prompts.json').read_text(encoding='utf-8'))
    records = []
    for probe in prompts:
        serialized = f"<|im_start|>user\n{probe['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        start = time.perf_counter()
        response = llm.generate([serialized], SamplingParams(**settings))[0]
        result = response.outputs[0]
        records.append(dict(probe=probe['id'], prompt=probe['prompt'],
            prompt_sha256=hashlib.sha256(probe['prompt'].encode()).hexdigest(),
            serialized_prompt=serialized, model=args.model, revision=args.revision,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            settings=settings, thinking=False, omics_profiles=0,
            response=result.text, finish_reason=result.finish_reason,
            generated_tokens=len(result.token_ids), prompt_tokens=len(response.prompt_token_ids),
            seconds=round(time.perf_counter()-start, 3),
            runtime={name: importlib.metadata.version(name) for name in ['torch','transformers','vllm','thalamus-reasoning']},
            gpu=torch.cuda.get_device_name(), python=platform.python_version()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
        print(probe['id'], result.finish_reason, len(result.token_ids), flush=True)


if __name__ == '__main__':
    main()
