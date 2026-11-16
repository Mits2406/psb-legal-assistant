"""Offline build of genuine BGE-M3 vectors. Never included in submission ZIP.

Requires torch==2.8.0, transformers==4.57.1 and the pinned public model files.
Runtime only needs stdlib and the organizer's embeddings API for query vectors.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import inspect
import json
from pathlib import Path
import time

REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
WEIGHTS_SHA256 = 'b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38'
PROBES = [
    'В какой срок необходимо оплатить поставленный товар?',
    'Согласие на обработку персональных данных можно отозвать.',
    'The rental agreement defines payment dates and access to the building.',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).with_name('data'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('precomputed_index.py'))
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps', 'auto'), default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--aux-texts', type=Path)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        parser.error('batch-size must be 1..64')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    checksum = sha256()
    with (args.model_dir / 'pytorch_model.bin').open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    if checksum.hexdigest() != WEIGHTS_SHA256:
        raise ValueError('Weights do not match the pinned BGE-M3 revision')

    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from corpus import Corpus
    from evidence import EvidencePlanner
    from llm import ModelClient
    from registry import RuleRegistry
    from retrieval import Retriever
    from vector_index import build_artifact, write_module, load_module

    corpus = Corpus(args.data_dir)
    planner = EvidencePlanner(corpus, RuleRegistry(corpus), ModelClient())
    inputs = {cid: planner.structure.contextual_text(chunk) for cid, chunk in planner.safe.items()}
    (args.work_dir / 'inputs.json').write_text(json.dumps(inputs, ensure_ascii=False), encoding='utf-8')
    aux = []
    demo = args.data_dir / 'demo_questions.csv'
    if demo.exists():
        with demo.open(encoding='utf-8-sig', newline='') as stream:
            aux.extend(row['question'] for row in csv.DictReader(stream))
    if args.aux_texts:
        aux.extend(json.loads(args.aux_texts.read_text(encoding='utf-8')))
    unique = list(dict.fromkeys([*inputs.values(), *PROBES, *aux]))
    device = args.device
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(6)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False,
                                     attn_implementation='sdpa').eval().to(device)
    lengths = {text: len(tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']) for text in unique}
    if max(lengths.values()) > 8192:
        raise ValueError('Some inputs exceed 8192 tokens; revise chunking before building an index')
    ordered = sorted(unique, key=lambda text: (lengths[text], text))
    print(json.dumps({'stage': 'encoding', 'device': device, 'chunks': len(inputs),
                      'unique_texts': len(unique), 'max_tokens': max(lengths.values())}), flush=True)
    encoded = {}
    with torch.inference_mode():
        for start in range(0, len(ordered), args.batch_size):
            texts = ordered[start:start + args.batch_size]
            tokens = tokenizer(texts, padding=True, truncation=False, return_tensors='pt').to(device)
            outputs = model(**tokens)
            vectors = torch.nn.functional.normalize(outputs.last_hidden_state[:, 0].float(), p=2, dim=1)
            for text, vector in zip(texts, vectors.cpu().tolist()):
                encoded[text] = vector
            if start % (args.batch_size * 5) == 0 or start + len(texts) == len(ordered):
                print(json.dumps({'encoded': len(encoded), 'total': len(ordered),
                                  'seconds': round(time.monotonic() - started, 1)}), flush=True)
    (args.work_dir / 'encoded_vectors.json').write_text(json.dumps(encoded, ensure_ascii=False), encoding='utf-8')
    chunker_hash = sha256()
    for name in ('corpus.py', 'chunk.py', 'document.py'):
        chunker_hash.update(Path(__file__).with_name(name).read_bytes())
    chunker_hash.update(inspect.getsource(Retriever.contextual_text).encode('utf-8'))
    artifact = build_artifact(
        inputs, {cid: encoded[text] for cid, text in inputs.items()}, model_revision=REVISION,
        corpus_hash=corpus.stats['corpus_hash'], chunker_fingerprint=chunker_hash.hexdigest(),
        weights_provenance={
            'repository': 'https://huggingface.co/BAAI/bge-m3', 'revision': REVISION,
            'filename': 'pytorch_model.bin', 'sha256': WEIGHTS_SHA256,
            'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
            'device': device, 'precision': 'float32', 'pooling': 'CLS', 'normalization': 'L2',
            'query_instruction': None, 'attention': 'sdpa', 'max_length': 8192,
            'truncated_inputs': 0, 'max_actual_tokens': max(lengths.values()),
        }, probes=[{'text': text, 'vector': encoded[text]} for text in PROBES])
    result = write_module(args.output, artifact)
    restored = load_module(args.output)
    matched, counters = restored.match_inputs(inputs)
    assert set(matched) == set(inputs), 'Index does not cover current corpus'
    report = {
        **result, 'documents': len(corpus.documents), 'all_chunks': len(corpus.chunks),
        'indexed_chunks': len(inputs), 'dimension': 1024, 'model': 'BAAI/bge-m3',
        'revision': REVISION, 'device': device, 'truncated_inputs': 0,
        'max_actual_tokens': max(lengths.values()), 'cache_validation': counters,
        'seconds': round(time.monotonic() - started, 2),
        'organizer_api_compatibility': 'not_tested; runtime probes required',
    }
    (args.work_dir / 'build_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
