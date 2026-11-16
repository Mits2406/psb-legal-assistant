"""Portable, content-addressed dense embeddings (runtime uses only the stdlib).

The generated ``precomputed_index.py`` is a literal data container, never imported
or executed. Its compressed payload contains a JSON manifest followed by a little
endian float32 matrix. Source text, source identity and applicability always come
from the *current* corpus; cached entries only supply vectors for exact input
hashes. A corpus hash records provenance and does not invalidate unchanged inputs.

The integrity digest detects accidental corruption, not a maliciously re-signed
artifact. Only trusted build output should be packaged with a submission.
"""
from __future__ import annotations

import ast
import base64
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct
import tempfile
import zlib


SCHEMA = 'psb-portable-dense-index-v1'
MODEL_ID = 'baai/bge-m3'
RECIPE_ID = 'bge-m3-cls-l2-v1'
DIMENSION = 1024
MAX_ROWS = 20_000
MAX_DIMENSION = 4096
MAX_RAW_BYTES = 96 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_MODULE_BYTES = 128 * 1024 * 1024
UNIT_TOLERANCE = 1e-4
_HASH = re.compile(r'[0-9a-f]{64}')


def input_hash(text: str) -> str:
    if not isinstance(text, str) or not text or len(text) > 1_000_000:
        raise ValueError('Embedding input must be a nonempty bounded string')
    return sha256(text.encode('utf-8')).hexdigest()


def normalize_model_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError('Invalid embedding model identity')
    return value.strip().lower()


def _hash(value, name):
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError('Invalid ' + name)
    return value


def _vector(values, dimension: int, *, normalize: bool) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != dimension:
        raise ValueError('Vector dimension mismatch')
    if any(type(x) not in (int, float) for x in values):
        raise ValueError('Invalid vector value')
    result = tuple(float(x) for x in values)
    if not all(math.isfinite(x) for x in result):
        raise ValueError('Nonfinite vector')
    length = math.hypot(*result)
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Zero or invalid vector norm')
    if normalize:
        return tuple(x / length for x in result)
    if abs(length - 1.0) > UNIT_TOLERANCE:
        raise ValueError('Cached vector is not normalized')
    return result


def _manifest_validate(manifest):
    if not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA:
        raise ValueError('Unsupported index schema')
    if manifest.get('model_id') != normalize_model_id(manifest.get('model_id')):
        raise ValueError('Noncanonical model identity')
    for key in ('recipe_id', 'model_revision', 'built_at'):
        if not isinstance(manifest.get(key), str) or not manifest[key] or len(manifest[key]) > 256:
            raise ValueError('Invalid ' + key)
    if manifest.get('storage') != 'little-endian-float32':
        raise ValueError('Unsupported vector storage')
    _hash(manifest.get('corpus_hash'), 'corpus hash')
    _hash(manifest.get('chunker_fingerprint'), 'chunker fingerprint')
    _hash(manifest.get('matrix_sha256'), 'matrix digest')
    dimension = manifest.get('dimension')
    if type(dimension) is not int or not 1 <= dimension <= MAX_DIMENSION:
        raise ValueError('Invalid index dimension')
    entries = manifest.get('input_hashes')
    if not isinstance(entries, list) or not 1 <= len(entries) <= MAX_ROWS:
        raise ValueError('Invalid index entries')
    for value in entries:
        _hash(value, 'input hash')
    if len(set(entries)) != len(entries) or entries != sorted(entries):
        raise ValueError('Duplicate or unsorted input hashes')
    if type(manifest.get('input_hash_count')) is not int or manifest['input_hash_count'] != len(entries):
        raise ValueError('Input hash count mismatch')
    probes = manifest.get('probes')
    if not isinstance(probes, list) or not 2 <= len(probes) <= 8:
        raise ValueError('Two to eight compatibility probes are required')
    probe_hashes = set()
    for item in probes:
        if not isinstance(item, dict) or set(item) != {'text', 'input_hash'}:
            raise ValueError('Invalid probe shape')
        digest = input_hash(item['text'])
        if digest != item['input_hash'] or digest in probe_hashes:
            raise ValueError('Invalid or duplicate probe input')
        probe_hashes.add(digest)
    provenance = manifest.get('weights_provenance')
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError('Weights provenance is required')
    try:
        encoded = json.dumps(provenance, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError('Invalid weights provenance') from None
    if len(encoded.encode()) > 100_000:
        raise ValueError('Oversized weights provenance')
    return dimension, entries, probes


class VectorIndex:
    """Validated float32 vectors plus portable metadata, without corpus objects."""

    def __init__(self, manifest: dict, matrix: bytes):
        dimension, entries, probes = _manifest_validate(manifest)
        expected = (len(entries) + len(probes)) * dimension * 4
        if not isinstance(matrix, bytes) or len(matrix) != expected or len(matrix) > MAX_RAW_BYTES:
            raise ValueError('Matrix length mismatch')
        if sha256(matrix).hexdigest() != manifest['matrix_sha256']:
            raise ValueError('Matrix digest mismatch')
        rows = []
        row_size = dimension * 4
        for offset in range(0, len(matrix), row_size):
            raw = struct.unpack_from('<' + str(dimension) + 'f', matrix, offset)
            rows.append(_vector(raw, dimension, normalize=False))
        # Freeze an independent manifest copy, so caller-owned nested objects
        # cannot alter the validated artifact between matching and writing.
        self.manifest = json.loads(json.dumps(manifest, ensure_ascii=False, allow_nan=False))
        self.dimension = dimension
        self._matrix = matrix
        self._vectors = dict(zip(entries, rows[:len(entries)]))
        self._probe_vectors = tuple(rows[len(entries):])
        self.probe_texts = tuple(item['text'] for item in probes)

    def match_inputs(self, current_inputs: Mapping[str, str]):
        """Return current IDs -> vectors and counts; no cached source IDs escape.

        ``reused`` and ``missing`` count current IDs; ``unused`` counts unique
        cached inputs absent from the current mapping. Thus a deleted document's
        vectors may remain in the artifact but can never be returned by matching.
        """
        if not isinstance(current_inputs, Mapping) or len(current_inputs) > MAX_ROWS:
            raise ValueError('Invalid current input map')
        result, current_hashes = {}, set()
        for current_id, text in current_inputs.items():
            if not isinstance(current_id, str) or not current_id:
                raise ValueError('Invalid current chunk ID')
            digest = input_hash(text)
            current_hashes.add(digest)
            if digest in self._vectors:
                result[current_id] = self._vectors[digest]
        return result, {
            'requested': len(current_inputs), 'reused': len(result),
            'missing': len(current_inputs) - len(result),
            'unused': len(set(self._vectors) - current_hashes),
            'unique_input_hashes': len(current_hashes),
        }

    def validate_probe_vectors(self, returned_vectors, threshold: float = .999):
        """Check provider space using all probes, before reusing offline vectors.

        Equal dimensions and a model alias alone are insufficient. This check
        helps detect different weights, pooling, truncation or preprocessing.
        It is a compatibility guard, not a proof of equivalent retrieval quality.
        """
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError('Invalid cosine threshold')
        report = {'compatible': False, 'cosines': [], 'threshold': threshold, 'reason': 'invalid_vectors'}
        if not isinstance(returned_vectors, (list, tuple)) or len(returned_vectors) != len(self._probe_vectors):
            return report
        try:
            returned = [_vector(row, self.dimension, normalize=True) for row in returned_vectors]
        except (ValueError, TypeError, OverflowError):
            return report
        cosines = [sum(x * y for x, y in zip(cached, actual))
                   for cached, actual in zip(self._probe_vectors, returned)]
        compatible = all(score >= threshold for score in cosines)
        return {'compatible': compatible, 'cosines': cosines, 'threshold': threshold,
                'reason': 'matched' if compatible else 'embedding_space_mismatch'}


def build_artifact(inputs: Mapping[str, str], vectors: Mapping[str, Sequence[float]], *,
                   model_id: str = MODEL_ID, model_revision: str, recipe_id: str = RECIPE_ID,
                   dimension: int = DIMENSION, corpus_hash: str, chunker_fingerprint: str,
                   weights_provenance: dict, probes: Sequence[dict], built_at: str | None = None) -> VectorIndex:
    """Build a validated artifact from *actual* encoder output; no model is called.

    ``inputs`` and ``vectors`` must have identical current chunk IDs. Probe items
    contain ``text`` and ``vector``. Vectors are L2-normalized then rounded to
    float32. Identical input strings deduplicate only if their vectors agree.
    """
    if not isinstance(inputs, Mapping) or not isinstance(vectors, Mapping) or set(inputs) != set(vectors):
        raise ValueError('Embedding input/vector coverage mismatch')
    if not 1 <= len(inputs) <= MAX_ROWS or type(dimension) is not int or not 1 <= dimension <= MAX_DIMENSION:
        raise ValueError('Invalid index size')
    if not isinstance(probes, (list, tuple)) or not 2 <= len(probes) <= 8:
        raise ValueError('Two to eight compatibility probes are required')
    by_hash = {}
    for current_id, text in inputs.items():
        if not isinstance(current_id, str) or not current_id:
            raise ValueError('Invalid current chunk ID')
        digest = input_hash(text)
        row = _vector(list(vectors[current_id]), dimension, normalize=True)
        previous = by_hash.get(digest)
        if previous is not None and any(abs(a - b) > 1e-5 for a, b in zip(previous, row)):
            raise ValueError('Same embedding input produced conflicting vectors')
        by_hash[digest] = row
    hashes = sorted(by_hash)
    probe_metadata, probe_rows = [], []
    for probe in probes:
        if not isinstance(probe, dict) or set(probe) != {'text', 'vector'}:
            raise ValueError('Probe requires text and vector')
        probe_metadata.append({'text': probe['text'], 'input_hash': input_hash(probe['text'])})
        probe_rows.append(_vector(list(probe['vector']), dimension, normalize=True))
    rows = [by_hash[key] for key in hashes] + probe_rows
    matrix = b''.join(struct.pack('<' + str(dimension) + 'f', *row) for row in rows)
    manifest = {
        'schema': SCHEMA, 'model_id': normalize_model_id(model_id),
        'model_revision': model_revision, 'recipe_id': recipe_id, 'dimension': dimension,
        'corpus_hash': corpus_hash, 'chunker_fingerprint': chunker_fingerprint,
        'built_at': built_at or datetime.now(timezone.utc).isoformat(),
        'weights_provenance': weights_provenance,
        'input_hash_count': len(hashes), 'input_hashes': hashes,
        'probes': probe_metadata, 'matrix_sha256': sha256(matrix).hexdigest(),
        'storage': 'little-endian-float32',
    }
    return VectorIndex(manifest, matrix)


def write_module(path: str | Path, artifact: VectorIndex) -> dict:
    """Atomically write a literal root .py file that survives platform packaging."""
    if not isinstance(artifact, VectorIndex):
        raise ValueError('A validated VectorIndex is required')
    # Revalidate public metadata in case a caller mutated ``manifest``.
    VectorIndex(artifact.manifest, artifact._matrix)
    manifest = json.dumps(artifact.manifest, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(manifest) > MAX_MANIFEST_BYTES:
        raise ValueError('Oversized manifest')
    raw = struct.pack('<I', len(manifest)) + manifest + artifact._matrix
    if len(raw) > MAX_RAW_BYTES:
        raise ValueError('Oversized index')
    encoded = base64.b85encode(zlib.compress(raw, level=9)).decode('ascii')
    digest = sha256(raw).hexdigest()
    lines = ['"""Generated embedding data. Read by vector_index; never execute model data."""',
             'INDEX_SCHEMA = ' + repr(SCHEMA), 'INDEX_SHA256 = ' + repr(digest), 'INDEX_B85 = (']
    lines.extend('    ' + repr(encoded[start:start + 100]) for start in range(0, len(encoded), 100))
    text = '\n'.join(lines + [')', ''])
    if len(text.encode()) > MAX_MODULE_BYTES:
        raise ValueError('Oversized generated module')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.index-', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'path': str(path), 'raw_bytes': len(raw), 'module_bytes': path.stat().st_size,
            'payload_sha256': digest, 'entries': artifact.manifest['input_hash_count']}


def load_module(path: str | Path, *, expected_model: str = MODEL_ID,
                expected_recipe: str = RECIPE_ID, expected_dimension: int = DIMENSION) -> VectorIndex:
    """Read literal data without importing/executing the generated Python module."""
    path = Path(path)
    if path.stat().st_size > MAX_MODULE_BYTES:
        raise ValueError('Oversized generated module')
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue  # optional module docstring
            if (not isinstance(node, ast.Assign) or len(node.targets) != 1
                    or not isinstance(node.targets[0], ast.Name)):
                raise ValueError('Generated index must contain literal assignments only')
            name = node.targets[0].id
            if name not in {'INDEX_SCHEMA', 'INDEX_SHA256', 'INDEX_B85'} or name in values:
                raise ValueError('Unexpected or duplicate index assignment')
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise ValueError('Index assignments must be strings')
            values[name] = value
        if set(values) != {'INDEX_SCHEMA', 'INDEX_SHA256', 'INDEX_B85'} or values['INDEX_SCHEMA'] != SCHEMA:
            raise ValueError('Missing or invalid index envelope')
        digest = _hash(values['INDEX_SHA256'], 'payload digest')
        compressed = base64.b85decode(values['INDEX_B85'])
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, MAX_RAW_BYTES + 1)
        if len(raw) > MAX_RAW_BYTES or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError('Invalid or oversized compressed payload')
        if sha256(raw).hexdigest() != digest or len(raw) < 4:
            raise ValueError('Payload digest mismatch')
        manifest_size = struct.unpack_from('<I', raw)[0]
        if not 0 < manifest_size <= MAX_MANIFEST_BYTES or manifest_size > len(raw) - 4:
            raise ValueError('Invalid manifest length')
        manifest = json.loads(raw[4:4 + manifest_size].decode('utf-8'))
        artifact = VectorIndex(manifest, raw[4 + manifest_size:])
        if artifact.manifest['model_id'] != normalize_model_id(expected_model):
            raise ValueError('Embedding model mismatch')
        if artifact.manifest['recipe_id'] != expected_recipe or artifact.dimension != expected_dimension:
            raise ValueError('Embedding recipe/dimension mismatch')
        return artifact
    except (SyntaxError, UnicodeError, TypeError, KeyError, OverflowError, zlib.error, struct.error, RecursionError) as error:
        raise ValueError('Invalid precomputed index: ' + type(error).__name__) from None
