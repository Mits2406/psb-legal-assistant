"""Portable-cache integrity and corpus-change tests; vectors are test fixtures only."""
import ast
import base64
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vector_index import build_artifact, load_module, write_module, VectorIndex


class PortableIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'precomputed_index.py'
        self.inputs = {'old-a': 'Документ: Договор А\nПункт: 4.2\nОплата через 30 дней.',
                       'old-b': 'Документ: Договор Б\nПункт: 6.1\nГарантия по талону.'}
        self.vectors = {'old-a': [1, 0, 0, 0], 'old-b': [0, 1, 0, 0]}
        self.options = dict(model_id='BAAI/bge-m3', model_revision='pinned-weights-test-only',
                            dimension=4, corpus_hash='a' * 64, chunker_fingerprint='b' * 64,
                            weights_provenance={'source': 'synthetic unit-test fixtures, not a production index'},
                            probes=[{'text': 'probe one', 'vector': [0, 0, 1, 0]},
                                    {'text': 'probe two', 'vector': [0, 0, 0, 1]}])

    def build(self):
        artifact = build_artifact(self.inputs, self.vectors, **self.options)
        write_module(self.path, artifact)
        return load_module(self.path, expected_dimension=4)

    def rewrite_payload(self, transform):
        values = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in ast.parse(self.path.read_text()).body if isinstance(node, ast.Assign)}
        raw = zlib.decompress(base64.b85decode(values['INDEX_B85']))
        size = struct.unpack_from('<I', raw)[0]
        manifest = json.loads(raw[4:4 + size])
        matrix = raw[4 + size:]
        manifest, matrix = transform(manifest, matrix)
        # Re-sign the envelope so inner structural/normalization validation must
        # catch corruption; a checksum-only loader would miss these cases.
        meta = json.dumps(manifest).encode()
        raw = struct.pack('<I', len(meta)) + meta + matrix
        values['INDEX_SHA256'] = sha256(raw).hexdigest()
        values['INDEX_B85'] = base64.b85encode(zlib.compress(raw)).decode()
        self.path.write_text('\n'.join(key + ' = ' + repr(value) for key, value in values.items()))

    def test_reuses_exact_input_under_new_source_id(self):
        index = self.build()
        vectors, counts = index.match_inputs({'renamed-id': self.inputs['old-a']})
        self.assertEqual(set(vectors), {'renamed-id'})
        self.assertEqual(vectors['renamed-id'], (1, 0, 0, 0))
        self.assertEqual(counts, {'requested': 1, 'reused': 1, 'missing': 0,
                                  'unused': 1, 'unique_input_hashes': 1})

    def test_changed_new_and_removed_chunks_only_reuse_exact_input(self):
        index = self.build()
        current = {'a': self.inputs['old-a'], 'changed-b': self.inputs['old-b'].replace('талону', 'договору'),
                   'new-c': 'Новое положение'}
        vectors, counts = index.match_inputs(current)
        self.assertEqual(set(vectors), {'a'})
        self.assertEqual((counts['reused'], counts['missing'], counts['unused']), (1, 2, 1))
        self.assertNotIn('old-b', vectors)

    def test_metadata_title_change_invalidates_vector(self):
        index = self.build()
        vectors, counts = index.match_inputs({'a': self.inputs['old-a'].replace('Договор А', 'Договор Б')})
        self.assertEqual(vectors, {})
        self.assertEqual(counts['missing'], 1)

    def test_hash_is_unicode_and_whitespace_exact(self):
        index = self.build()
        vectors, counts = index.match_inputs({'a': self.inputs['old-a'] + ' '})
        self.assertFalse(vectors)
        self.assertEqual(counts['unused'], 2)

    def test_duplicate_inputs_reuse_one_vector_for_current_ids(self):
        self.inputs['old-c'] = self.inputs['old-a']
        self.vectors['old-c'] = self.vectors['old-a']
        index = self.build()
        self.assertEqual(index.manifest['input_hash_count'], 2)
        vectors, counts = index.match_inputs({'x': self.inputs['old-a'], 'y': self.inputs['old-a']})
        self.assertEqual(set(vectors), {'x', 'y'})
        self.assertEqual(counts['unique_input_hashes'], 1)

    def test_conflicting_duplicate_input_vectors_rejected(self):
        self.inputs['old-c'] = self.inputs['old-a']
        self.vectors['old-c'] = self.vectors['old-b']
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            build_artifact(self.inputs, self.vectors, **self.options)

    def test_model_case_is_normalized_but_other_model_rejected(self):
        self.build()
        self.assertEqual(load_module(self.path, expected_model='BaAi/BGE-M3', expected_dimension=4).dimension, 4)
        with self.assertRaisesRegex(ValueError, 'model mismatch'):
            load_module(self.path, expected_model='different/model', expected_dimension=4)

    def test_recipe_and_dimension_mismatch_rejected(self):
        self.build()
        with self.assertRaisesRegex(ValueError, 'recipe/dimension'):
            load_module(self.path, expected_recipe='different', expected_dimension=4)
        with self.assertRaisesRegex(ValueError, 'recipe/dimension'):
            load_module(self.path, expected_dimension=1024)

    def test_probe_compatibility_checks_every_probe_and_vector_space(self):
        index = self.build()
        self.assertTrue(index.validate_probe_vectors([[0, 0, 2, 0], [0, 0, 0, 3]])['compatible'])
        failed = index.validate_probe_vectors([[0, 0, 1, 0], [1, 0, 0, 0]])
        self.assertFalse(failed['compatible'])
        self.assertEqual(failed['reason'], 'embedding_space_mismatch')
        self.assertEqual(failed['cosines'], [1, 0])
        self.assertFalse(index.validate_probe_vectors([[1, 0, 0, 0]])['compatible'])
        self.assertFalse(index.validate_probe_vectors([[0, 0, 1, 0], [0, 0, float('nan'), 1]])['compatible'])

    def test_nonfinite_and_zero_build_vectors_rejected(self):
        for row in ([0, 0, 0, 0], [float('nan'), 0, 1, 0], [float('inf'), 1, 0, 0]):
            self.vectors['old-a'] = row
            with self.assertRaises(ValueError):
                build_artifact(self.inputs, self.vectors, **self.options)

    def test_corrupted_payload_digest_rejected(self):
        self.build()
        text = self.path.read_text()
        text = text.replace('INDEX_SHA256 = ', 'INDEX_SHA256 = ' + repr('0' * 64) + '\n# ', 1)
        self.path.write_text(text)
        with self.assertRaisesRegex(ValueError, 'digest'):
            load_module(self.path, expected_dimension=4)

    def test_inner_matrix_digest_rejected_even_if_envelope_recomputed(self):
        self.build()
        self.rewrite_payload(lambda meta, matrix: (meta, bytes([matrix[0] ^ 1]) + matrix[1:]))
        with self.assertRaisesRegex(ValueError, 'Matrix digest'):
            load_module(self.path, expected_dimension=4)

    def test_nonunit_matrix_rejected_even_with_valid_digests(self):
        self.build()
        def transform(meta, matrix):
            modified = struct.pack('<4f', 2, 0, 0, 0) + matrix[16:]
            meta['matrix_sha256'] = sha256(modified).hexdigest()
            return meta, modified
        self.rewrite_payload(transform)
        with self.assertRaisesRegex(ValueError, 'not normalized'):
            load_module(self.path, expected_dimension=4)

    def test_nonfinite_matrix_rejected_even_with_valid_digests(self):
        self.build()
        def transform(meta, matrix):
            modified = struct.pack('<4f', float('nan'), 0, 0, 0) + matrix[16:]
            meta['matrix_sha256'] = sha256(modified).hexdigest()
            return meta, modified
        self.rewrite_payload(transform)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            load_module(self.path, expected_dimension=4)

    def test_module_code_is_rejected_and_never_executed(self):
        self.build()
        sentinel = Path(self.temp.name) / 'executed'
        with self.path.open('a') as handle:
            handle.write('\n__import__("pathlib").Path(' + repr(str(sentinel)) + ').touch()\n')
        with self.assertRaisesRegex(ValueError, 'literal assignments'):
            load_module(self.path, expected_dimension=4)
        self.assertFalse(sentinel.exists())

    def test_size_caps_reject_module_and_decompressed_payload(self):
        self.build()
        with patch('vector_index.MAX_MODULE_BYTES', 16):
            with self.assertRaisesRegex(ValueError, 'Oversized generated module'):
                load_module(self.path, expected_dimension=4)
        # The bound is applied during decompression, before trusting a manifest.
        with patch('vector_index.MAX_RAW_BYTES', 16):
            with self.assertRaisesRegex(ValueError, 'oversized compressed payload'):
                load_module(self.path, expected_dimension=4)

    def test_truncated_or_extra_compressed_stream_is_rejected(self):
        self.build()
        values = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in ast.parse(self.path.read_text()).body if isinstance(node, ast.Assign)}
        compressed = base64.b85decode(values['INDEX_B85'])
        for broken in (compressed[:-1], compressed + zlib.compress(b'unexpected stream')):
            with self.subTest(length=len(broken)):
                values['INDEX_B85'] = base64.b85encode(broken).decode()
                self.path.write_text('\n'.join(key + ' = ' + repr(value) for key, value in values.items()))
                with self.assertRaisesRegex(ValueError, 'compressed payload'):
                    load_module(self.path, expected_dimension=4)

    def test_manifest_rejects_unknown_storage_and_invalid_hash_shape(self):
        for key, value in (('storage', 'big-endian-float32'), ('input_hashes', [{}]),
                           ('input_hash_count', True)):
            with self.subTest(key=key):
                self.build()
                def transform(meta, matrix):
                    meta[key] = value
                    return meta, matrix
                self.rewrite_payload(transform)
                with self.assertRaises(ValueError):
                    load_module(self.path, expected_dimension=4)

    def test_missing_vector_and_duplicate_probe_rejected(self):
        self.vectors.pop('old-b')
        with self.assertRaisesRegex(ValueError, 'coverage'):
            build_artifact(self.inputs, self.vectors, **self.options)
        self.vectors['old-b'] = [0, 1, 0, 0]
        self.options['probes'][1]['text'] = self.options['probes'][0]['text']
        with self.assertRaisesRegex(ValueError, 'duplicate probe'):
            build_artifact(self.inputs, self.vectors, **self.options)


if __name__ == '__main__':
    unittest.main()
