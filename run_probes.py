"""Run the exact three prompts against an OpenAI-compatible model endpoint.

Uses only Python's standard library. Credentials stay in MODEL_API_KEY, never
in artifacts. Arguments are explicit so different model runs can be compared.
No network calls are made by demo.py. No model is called by default.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', required=True, help='Full /v1/chat/completions URL')
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True, help='Checkpoint hash or dated provider model ID')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--max-tokens', type=int, default=1536)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output path; raw runs are immutable.')
    if not args.endpoint.startswith(('https://', 'http://127.0.0.1:', 'http://localhost:')):
        p.error('Use HTTPS or an explicit loopback server.')
    prompts = json.loads(Path(__file__).with_name('prompts.json').read_text(encoding='utf-8'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        for probe in prompts:
            payload = dict(model=args.model, messages=[dict(role='user', content=probe['prompt'])],
                           temperature=0, max_tokens=args.max_tokens)
            headers = {'Content-Type': 'application/json'}
            if os.getenv('MODEL_API_KEY'):
                headers['Authorization'] = 'Bearer ' + os.environ['MODEL_API_KEY']
            start = time.perf_counter()
            req = Request(args.endpoint, data=json.dumps(payload).encode(), headers=headers, method='POST')
            with urlopen(req, timeout=600) as response:
                raw = json.loads(response.read())
            record = dict(probe=probe['id'], prompt=probe['prompt'],
                          prompt_sha256=hashlib.sha256(probe['prompt'].encode()).hexdigest(),
                          requested_model=args.model, revision=args.revision,
                          timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
                          seconds=round(time.perf_counter()-start, 3),
                          settings={'temperature': 0, 'max_tokens': args.max_tokens}, response=raw)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            print(probe['id'], 'saved')


if __name__ == '__main__':
    main()
