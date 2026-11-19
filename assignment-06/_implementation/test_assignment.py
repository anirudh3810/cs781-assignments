"""Deterministic unit/contract tests. Fixtures are never assignment results."""
import ast
import contextlib
import copy
import io
import json
import pathlib
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import doaa_shared as s
import numpy as np
import faiss
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, List, Annotated
import operator

NB = json.loads((ROOT / 'Assignment6_RAGandAgenticAI_Questions_Notebook.ipynb').read_text(encoding='utf-8'))
ORIG = json.loads((ROOT / '.backups/original/Assignment6_RAGandAgenticAI_Questions_Notebook.ipynb').read_text(encoding='utf-8'))
BY_ID = {cell['metadata']['id']: cell for cell in NB['cells']}


def original_code(index):
    return ''.join(BY_ID[ORIG['cells'][index]['metadata']['id']]['source'])


def definitions(index, namespace):
    tree = ast.parse(original_code(index))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<actual notebook definitions>', 'exec'), namespace)


class CoreTests(unittest.TestCase):
    def test_student_review_requires_confirmation_and_input_binding(self):
        supplied = {'evaluation_input_hash': 'fixture-hash', 'evaluator': 'Unit reviewer',
                    'review_method': 'human_reviewer', 'reviewer_confirmed': True,
                    'independent_human_review': False, 'rating_origin': 'Explicitly confirmed unit fixture',
                    'scores': {'0': {'relevance': 3, 'faithfulness': 2, 'completeness': 4}}}
        rows, provenance = s.validate_human_review(supplied, 'fixture-hash', 1)
        self.assertEqual(rows[0]['completeness'], 4)
        self.assertFalse(provenance['independent_human_review'])
        self.assertTrue(provenance['reviewer_confirmed'])
        for mutation in ({'independent_human_review': True}, {'reviewer_confirmed': False},
                         {'rating_origin': ''}, {'evaluation_input_hash': 'stale'}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                s.validate_human_review({**supplied, **mutation}, 'fixture-hash', 1)

    def test_independent_review_rejects_invalid_scores(self):
        supplied = {'evaluation_input_hash': 'fixture-hash', 'evaluator': 'Unit reviewer',
                    'independent_human_review': True,
                    'scores': {'0': {'relevance': 3, 'faithfulness': 2, 'completeness': 4}}}
        self.assertTrue(s.validate_human_review(supplied, 'fixture-hash', 1)[1]['independent_human_review'])
        for value in (True, 6, -1, 3.5, '4'):
            bad = copy.deepcopy(supplied); bad['scores']['0']['relevance'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                s.validate_human_review(bad, 'fixture-hash', 1)

    def test_fixed_chunks_and_invalid_sizes(self):
        for text, size, expected in [('', 2, []), ('abcd', 2, ['ab', 'cd']), ('abcde', 2, ['ab', 'cd', 'e']),
                                     ('αβγ', 2, ['αβ', 'γ']), ('   ', 2, ['  ', ' '])]:
            with self.subTest(text=text):
                self.assertEqual(s.chunk_text_core(text, size), expected)
        for size in (0, -1, 1.5, True, None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                s.chunk_text_core('abc', size)
        with self.assertRaises(TypeError):
            s.chunk_text_core(None, 2)

    def test_url_scope_and_queries(self):
        self.assertEqual(s.normalize_doaa_url('/doaa/a.PDF?x=1#p'), 'https://www.iitk.ac.in/doaa/a.PDF?x=1')
        self.assertEqual(s.normalize_doaa_url('file.pdf'), 'https://www.iitk.ac.in/doaa/file.pdf')
        for invalid in ('', '#x', 'mailto:x@y.com', 'javascript:1', '/doaa-other', 'https://iitk.ac.in.evil/doaa/a',
                        'https://user:pass@www.iitk.ac.in/doaa/', 'https://www.iitk.ac.in:bad/doaa/'):
            with self.subTest(url=invalid):
                self.assertIsNone(s.normalize_doaa_url(invalid))

    def test_html_extraction_actual_parser(self):
        result = s.extract_html(b'<html><title>Fixture</title><body><nav>boilerplate</nav><main><h1>Policy</h1><p>Credit: 4</p><script>bad()</script><a href="/doaa/rules.pdf">Rules</a></main></body></html>', s.BASE_URL)
        text = result['pages'][0]['text']
        self.assertIn('Policy', text)
        self.assertIn('Credit: 4', text)
        self.assertNotIn('bad()', text)
        self.assertNotIn('boilerplate', text)
        self.assertEqual(result['links'], [s.BASE_URL + 'rules.pdf'])

    def test_pdf_actual_reader(self):
        # Construct a tiny genuine text PDF solely as a parser unit fixture.
        contents = b'BT /F1 12 Tf 10 100 Td (PDF extraction unit fixture) Tj ET'
        objects = [b'<< /Type /Catalog /Pages 2 0 R >>', b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
                   b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
                   b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
                   b'<< /Length ' + str(len(contents)).encode() + b' >>\nstream\n' + contents + b'\nendstream']
        body = b'%PDF-1.4\n'
        offsets = [0]
        for number, obj in enumerate(objects, 1):
            offsets.append(len(body))
            body += str(number).encode() + b' 0 obj\n' + obj + b'\nendobj\n'
        xref = len(body)
        body += b'xref\n0 6\n0000000000 65535 f \n' + b''.join(f'{offset:010} 00000 n \n'.encode() for offset in offsets[1:])
        body += f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF'.encode()
        self.assertIn('PDF extraction unit fixture', s.extract_pdf(body)['pages'][0]['text'])
        with self.assertRaises(Exception):
            s.extract_pdf(b'not a PDF')

    def test_spi_calculation_and_edge_cases(self):
        self.assertIn('8.18', s.spi_core('4:9, 3:8, 4:10, 3:7, 3:6'))
        self.assertIn('8.64', s.spi_core('4:9,3:8,4:10,3:7'))
        self.assertIn('5.00', s.spi_core('2:0,2:10'))
        self.assertIn('8.50', s.spi_core('1.5:8,1.5:9'))
        for bad in ('', '4', '4:11', '4:-1', '0:8', '-1:8', 'nan:8', '4:inf', '4:9,', '1:2:3', None):
            with self.subTest(bad=bad):
                self.assertTrue(s.spi_core(bad).startswith('Error:'))

    def test_atomic_cache_corruption_and_binary(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.a6_artifacts') as folder:
            store = s.ArtifactStore(folder)
            self.assertIsNone(store.load('unit', 'missing'))
            store.save('unit', 'value', {'x': 3})
            self.assertEqual(store.load('unit', 'value'), {'x': 3})
            envelope = json.loads(store.path('unit', 'value').read_text(encoding='utf8'))
            envelope['payload']['x'] = 4
            store.path('unit', 'value').write_text(json.dumps(envelope), encoding='utf8')
            self.assertIsNone(store.load('unit', 'value'))
            store.save_blob('unit', 'binary', b'valid bytes')
            self.assertEqual(store.load_blob('unit', 'binary'), b'valid bytes')
            descriptor = store.load('unit', 'binary')
            (store.path('unit', 'binary').parent / descriptor['file']).write_bytes(b'corrupt')
            self.assertIsNone(store.load_blob('unit', 'binary'))
            with self.assertRaises(ValueError):
                store.path('../escape', 'x')

    def test_normalization_and_faiss(self):
        vectors = s.normalized_f32([[3, 4], [4, 3]], 2)
        self.assertTrue(vectors.flags.c_contiguous)
        self.assertEqual(vectors.dtype, np.float32)
        self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1))
        index = faiss.IndexFlatIP(2)
        index.add(vectors)
        scores, ids = index.search(vectors[:1], 3)
        self.assertEqual(ids[0, 0], 0)
        self.assertEqual(ids[0, -1], -1)
        for wrong in ([[0, 0]], [[float('nan'), 1]], [[float('inf'), 1]]):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                s.normalized_f32(wrong, 2)
        with self.assertRaises(ValueError):
            s.normalized_f32([[1, 2]], 384)

    def test_retrieval_filters_and_negative_indices(self):
        class Encoder:
            def encode(self, *args, **kwargs):
                return np.array([1, 0], dtype=np.float32)
        vectors = s.normalized_f32([[1, 0], [0, 1]], 2)
        index = faiss.IndexFlatIP(2)
        index.add(vectors)
        chunks = ['PhD thesis submission steps', '2026 semester exam calendar']
        self.assertEqual(len(s.retrieve_records('PhD', Encoder(), index, chunks, k=20)), 2)
        self.assertEqual(s.retrieve_records('PhD', Encoder(), index, chunks, kind='thesis')[0]['chunk_id'], 0)
        self.assertEqual(s.retrieve_records('2026 exams', Encoder(), index, chunks, kind='schedule')[0]['chunk_id'], 1)
        self.assertEqual(s.retrieve_records('2027 exams', Encoder(), index, chunks, kind='schedule'), [])
        with self.assertRaises(ValueError):
            s.retrieve_records('', Encoder(), index, chunks)
        with self.assertRaises(ValueError):
            s.retrieve_records('x', Encoder(), index, chunks[:1])

    def test_chunk_cross_document_provenance(self):
        chunks = s.chunk_text_core('abc\n\ndef', 5)
        spans = [{'start': 0, 'end': 3, 'url': 'a'}, {'start': 5, 'end': 8, 'url': 'b'}]
        metadata = s.chunk_metadata_core(chunks, spans)
        self.assertEqual(metadata[0]['sources'][0]['url'], 'a')
        self.assertEqual(metadata[1]['sources'][0]['url'], 'b')
        cross = s.chunk_metadata_core(['abcdef'], [{'start': 0, 'end': 3, 'url': 'a'}, {'start': 3, 'end': 6, 'url': 'b'}])
        self.assertEqual(len(cross[0]['sources']), 2)

    def test_judge_parser_strictness(self):
        for number in range(1, 6):
            self.assertEqual(s.parse_judge_response(f' SCORE: {number}\nREASON: evidence ')[0], number)
            self.assertEqual(s.parse_judge_response(f'**Score:** {number}\n**Reason:** evidence')[0], number)
            self.assertEqual(s.parse_judge_response(f'**SCORE**: {number}\nReason: evidence')[0], number)
        for bad in ('SCORE: 0\nREASON: x', 'SCORE: 6\nREASON: x', 'SCORE: 3.5\nREASON: x',
                    'SCORE: 3\nREASON:', 'Rating is 4', 'SCORE: 3\nREASON: x\nSCORE: 4\nREASON: y'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                s.parse_judge_response(bad)

    def test_real_tool_call_parser_not_invented(self):
        schemas = [{'type': 'function', 'function': {'name': 'calc', 'parameters': {'type': 'object',
                   'properties': {'grades': {'type': 'string'}}, 'required': ['grades'], 'additionalProperties': False}}}]
        good = '<tool_call>{"name":"calc","arguments":{"grades":"4:9"}}</tool_call>'
        self.assertEqual(s.parse_tool_calls(good, schemas)[0]['args'], {'grades': '4:9'})
        self.assertEqual(s.parse_tool_calls('A normal final answer.', schemas), [])
        for bad in ('<tool_call>{"name":"evil","arguments":{}}</tool_call>',
                    '<tool_call>{"name":"calc","arguments":{"grades":3}}</tool_call>', '<tool_call>{'):
            with self.subTest(bad=bad), self.assertRaises(Exception):
                s.parse_tool_calls(bad, schemas)

    def test_agreement_mathematics_no_fake_empirical_scores(self):
        human = {0: {'x': 5}, 1: {'x': 4}, 2: {'x': 3}, 3: {'x': 0}}
        judge = [{'x': 5}, {'x': 3}, {'x': 1}, {'x': 5}]
        result = s.agreement_counts(human, judge, ['x'])
        self.assertEqual((result['exact'], result['close'], result['total']), (1, 2, 3))
        self.assertIsNone(s.agreement_counts({0: {'x': 0}}, [{'x': 3}], ['x'])['exact_percent'])
        with self.assertRaises(ValueError):
            s.agreement_counts({0: {'x': 6}}, [{'x': 3}], ['x'])

    def test_notebook_preservation_images_and_alternative(self):
        original_ids = [c['metadata']['id'] for c in ORIG['cells']]
        self.assertEqual([c['metadata']['id'] for c in NB['cells'] if c['metadata']['id'] in original_ids], original_ids)
        self.assertEqual(len(NB['cells']), 262)
        for old in ORIG['cells']:
            new = BY_ID[old['metadata']['id']]
            self.assertEqual(old['cell_type'], new['cell_type'])
            import re
            self.assertEqual(re.findall(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', ''.join(old['source'])),
                             re.findall(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', ''.join(new['source'])))
        for index in (201, 203, 205, 207, 208, 210, 212, 214, 216, 218, 220, 222):
            wrapped = original_code(index)
            inactive = textwrap.dedent(wrapped.split('\nelse:\n')[0].split('\n', 1)[1]).strip()
            self.assertEqual(inactive, ''.join(ORIG['cells'][index]['source']).strip())

    def test_notebook_tools_are_actual_decorated_interfaces(self):
        ns = {'tool': tool, 'shared': s, 'safe_search': lambda query, kind=None: 'Error: blank query' if not query else 'Fixture evidence'}
        helper_tree = ast.parse((ROOT / '_implementation/h1.py').read_text(encoding='utf8'))
        protect = next(node for node in helper_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'protect_tool')
        exec(compile(ast.Module(body=[protect], type_ignores=[]), '<actual tool input guard>', 'exec'), ns)
        for index in (101, 104, 107, 110):
            definitions(index, ns)
            tree = ast.parse(original_code(index))
            for node in tree.body:
                if isinstance(node, ast.Assign) and (any(isinstance(target, ast.Attribute) for target in node.targets)
                        or (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'protect_tool')):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), '<tool error configuration>', 'exec'), ns)
        for name in ('search_doaa_knowledge', 'calculate_spi', 'lookup_thesis_procedure', 'lookup_schedule'):
            obj = ns[name]
            self.assertTrue(hasattr(obj, 'invoke'))
            self.assertTrue(obj.invoke({'bad': 'input'}).startswith('Error:'))
            self.assertTrue(obj.invoke(123).startswith('Error:'))
        self.assertIn('8.18', ns['calculate_spi'].invoke('4:9,3:8,4:10,3:7,3:6'))

    def test_notebook_public_signatures_and_routers(self):
        import inspect
        ns = {'shared': s, 'AgentState': dict, 'MAX_AGENT_ATTEMPTS': 3,
              '_run_qwen_impl': lambda message, tokens: str(tokens)}
        for index, default in ((60, 500), (94, 256)):
            definitions(index, ns)
            self.assertEqual(inspect.signature(ns['run_qwen']).parameters['max_tokens'].default, default)
            self.assertEqual(ns['run_qwen']('fixture', 17), '17')
        definitions(131, ns)
        self.assertEqual(ns['route_after_reasoning']({'next_action': 'tool'}), 'tool_executor')
        self.assertEqual(ns['route_after_quality_check']({'next_action': 'retry', 'iteration': 3}), 'end')
        with self.assertRaises(ValueError):
            ns['route_after_reasoning']({'next_action': 'unknown'})

    def test_all_original_function_argument_contracts(self):
        for original_index, cell in enumerate(ORIG['cells']):
            if cell['cell_type'] != 'code' or 201 <= original_index <= 222:
                continue
            try:
                old_tree = ast.parse(''.join(cell['source']))
            except SyntaxError:
                continue
            public = [node for node in old_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if not public:
                continue
            new_tree = ast.parse(original_code(original_index))
            current = {node.name: node for node in new_tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for function in public:
                with self.subTest(original_cell=original_index, function=function.name):
                    self.assertIn(function.name, current)
                    self.assertEqual(ast.dump(function.args), ast.dump(current[function.name].args))

    def test_cutoff_answer_retries_even_if_model_accepts(self):
        def last_question(messages):
            return next(m.content for m in reversed(messages) if isinstance(m, HumanMessage))
        import re
        namespace = {'AgentState': dict, 'AIMessage': AIMessage, 'HumanMessage': HumanMessage, 're': re,
                     'latest_human_question': last_question, 'parse_json_object': s.parse_json_object,
                     'trace_event': lambda *args, **kwargs: None, 'MAX_AGENT_ATTEMPTS': 3, 'shared': s,
                     'run_qwen': lambda *args, **kwargs: '{"accept":true,"reason":"Synthetic test acceptance"}'}
        definitions(128, namespace)
        state = {'messages': [HumanMessage(content='Synthetic quality regression fixture'),
                              AIMessage(content='An unfinished sentence', additional_kwargs={'generation_limit_reached': True})],
                 'tool_result': '', 'final_answer': 'An unfinished sentence', 'iteration': 1}
        result = namespace['quality_checker'](state)
        self.assertEqual(result['next_action'], 'retry')
        self.assertIn('cut off', result['messages'][0].content)
        state['iteration'] = 3
        result = namespace['quality_checker'](state)
        self.assertEqual(result['next_action'], 'accept')
        self.assertIn('Limitation:', result['final_answer'])

        state.update(iteration=1, tool_result='Synthetic actual-question evidence', final_answer='Quality feedback: This revision maintains the word count.')
        state['messages'][-1] = AIMessage(content=state['final_answer'])
        self.assertEqual(namespace['quality_checker'](state)['next_action'], 'retry')
        namespace['run_qwen'] = lambda *args, **kwargs: '{"accept":true}'
        state['final_answer'] = 'A complete source-supported answer.'
        state['messages'][-1] = AIMessage(content=state['final_answer'])
        self.assertEqual(namespace['quality_checker'](state)['next_action'], 'accept')
        with self.assertRaises(ValueError):
            s.parse_judge_response('{"accept":true}')  # Compulsory 1–5 judgments still require a score and reason.

    def test_truncation_uses_budget_and_structural_evidence(self):
        limited = {'output_tokens': 384}
        self.assertTrue(s.generation_truncation_issues('Submit a PDF of their', limited, 384))
        self.assertTrue(s.generation_truncation_issues('A required document (including the form', limited, 384))
        self.assertTrue(s.generation_truncation_issues('Complete step.\n2. ', limited, 384))
        for ending in ('Complete claim [C123]', 'Total credits (17)', '3. Total credits: 17', 'Complete claim.'):
            self.assertEqual(s.generation_truncation_issues(ending, limited, 384), [])
        self.assertEqual(s.generation_truncation_issues('A short heading', {'output_tokens': 7}, 384), [])

    def test_calendar_period_filter_beats_wrong_year_semantic_match(self):
        class Encoder:
            def encode(self, *args, **kwargs):
                return np.array([1, 0], dtype=np.float32)
        chunks = ['Odd semester 2033-34-I mid semester examination. Also mentions 2034.',
                  'Academic Calendar 2034\nEVENTS/ACTIVITIES SEMESTER 2033-34-II SUMMER TERM 2034 SEMESTER 2034-35-I',
                  'Mid Semester Examination\nFeb 20-25, 2034\nJune 12-14, 2034\nSep 15-21, 2034']
        metadata = [{'sha256': s.digest(c), 'sources': [{'url': s.BASE_URL + 'data/Calendar-' + ('2033' if i == 0 else '2034') + '.pdf'}]} for i,c in enumerate(chunks)]
        index = faiss.IndexFlatIP(2)
        index.add(s.normalized_f32([[1,0],[0.1,1],[0.2,1]],2))
        records = s.retrieve_records('When are mid-semester exams for odd semester 2034?', Encoder(), index, chunks, metadata, kind='schedule')
        self.assertEqual([r['chunk_id'] for r in records], [2,1])
        self.assertEqual(records[0]['calendar_request']['academic_period'], '2034-35-I')
        event = s.verified_calendar_event('When are mid-semester exams for odd semester 2034?', records)
        self.assertIsNotNone(event)
        self.assertEqual(event['date_range'], 'Sep 15-21, 2034')
        self.assertTrue(s.format_schedule_records('When are mid-semester exams for odd semester 2034?', records).startswith('Verified calendar event:'))
        self.assertEqual(s.retrieve_records('When are exams for odd semester 2035?', Encoder(), index, chunks, metadata, kind='schedule'), [])

    def test_temporal_contradiction_and_explicit_historical_context(self):
        question = 'When are the mid-semester exams for odd semester 2026?'
        self.assertTrue(s.temporal_grounding_issues(question, 'The mid-semester exam for odd semester 2026 falls on Tuesday, 29th October 2025.', 'Calendar evidence'))
        self.assertEqual(s.temporal_grounding_issues(question, 'The historical 2025 calendar does not verify the requested 2026 exam date.', 'Calendar evidence'), [])
        self.assertTrue(s.temporal_grounding_issues(question, 'Exams are on 21 February 2026.', 'No matching evidence was found.'))

    def test_source_sentence_selection_rejects_invented_claims(self):
        records = [{'chunk_id': 7, 'text': 'A source statement is complete.\n• Another requirement is documented.\nUnfinished phrase about their', 'sources': []}]
        catalog = s.evidence_sentence_catalog(records)
        self.assertEqual(len(catalog), 2)
        self.assertEqual(s.parse_sentence_selection('[1,2]', catalog), [1,2])
        for invalid in ('[3]', '[1,1]', '[true]', '["1"]', '{"ids":[1]}'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                s.parse_sentence_selection(invalid, catalog)
        self.assertTrue(all(row['text'].endswith('.') for row in catalog))

    def test_evidence_scope_guard_avoids_fee_formula_and_historical_deadline(self):
        certificate = [{'chunk_id': 4, 'text': 'CPI conversion certificate. Processing fee is Rs. 150.', 'sources': [{'url': s.BASE_URL+'certificate.pdf'}]}]
        self.assertIsNotNone(s.evidence_scope_limit('How is CPI calculated?', certificate))
        self.assertIsNone(s.evidence_scope_limit('What is the CPI conversion certificate fee?', certificate))
        minutes = [{'chunk_id': 5, 'text': 'The committee considered an individual extension request in 2015.', 'sources': [{'url': s.BASE_URL+'data/SPGC-201415-12.pdf'}]}]
        self.assertIsNotNone(s.evidence_scope_limit('What is the general seminar deadline?', minutes))

    def test_sentence_provenance_excludes_unrelated_boundary_document(self):
        unrelated = 'Unrelated refunds are discussed in this document. '
        text = unrelated + 'Students must submit the thesis form. The Head must sign the thesis form.'
        records = [{'chunk_id': 10, 'text': text, 'sources': [
            {'url': s.BASE_URL+'Refund_FAQ.pdf', 'chunk_start': 0, 'chunk_end': len(unrelated)},
            {'url': s.BASE_URL+'PhD_Thesis_Submission_Procedure.pdf', 'chunk_start': len(unrelated), 'chunk_end': len(text)}]}]
        catalog = s.evidence_sentence_catalog(records)
        selected = s.prioritize_evidence_sentences('What is the PhD thesis submission procedure?', catalog)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all('refund' not in row['text'].lower() for row in selected))
        self.assertEqual([row['id'] for row in selected], [1,2])

    def test_sentence_catalog_preserves_scope_of_contextual_clauses(self):
        text = ('Adding courses is not permitted in the summer term. However, students may drop a course until a specified deadline. '
                'Total credits should not exceed 44. This includes Summer terms.')
        catalog = s.evidence_sentence_catalog([{'chunk_id': 0, 'text': text, 'sources': []}])
        drops = [row['text'] for row in catalog if 'drop a course' in row['text']]
        self.assertTrue(drops)
        self.assertTrue(all('summer term' in sentence for sentence in drops))
        credits = [row['text'] for row in catalog if '44' in row['text']]
        self.assertTrue(all('Summer terms' in sentence for sentence in credits))

    def test_smoke_memory_archive_preserves_full_conversation(self):
        helper_tree = ast.parse((ROOT / '_implementation/h1.py').read_text(encoding='utf8'))
        nodes = [node for node in helper_tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'conversation_key', 'save_conversation'}]
        with tempfile.TemporaryDirectory(dir=ROOT / '.a6_artifacts') as folder:
            store = s.ArtifactStore(folder)
            namespace = {'STORE': store, 'digest': s.digest, 'RUN_MODE': 'full',
                         'corpus': {'corpus_hash': 'synthetic-full-corpus'}, 'GRAPH_IMPLEMENTATION_KEY': 'synthetic-graph'}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual scoped conversation archive>', 'exec'), namespace)
            namespace['save_conversation']({'messages': [HumanMessage(content='Synthetic full conversation')]}, 'same-thread')
            full_key = namespace['conversation_key']('same-thread')
            full_record = store.load('conversations', full_key)
            namespace['RUN_MODE'] = 'smoke'
            namespace['corpus'] = {'corpus_hash': 'synthetic-smoke-corpus'}
            namespace['save_conversation']({'messages': [HumanMessage(content='Synthetic smoke conversation')]}, 'same-thread')
            self.assertNotEqual(full_key, namespace['conversation_key']('same-thread'))
            self.assertEqual(store.load('conversations', full_key), full_record)

    def test_published_snapshot_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory(dir=ROOT / '.a6_artifacts') as folder:
            folder = pathlib.Path(folder)
            text = 'abcdef'
            chunks = ['abc', 'def']
            spans = [{'start': 0, 'end': 6, 'url': s.BASE_URL + 'unit-fixture', 'title': 'Synthetic unit fixture', 'page': None, 'kind': 'html'}]
            metadata = s.chunk_metadata_core(chunks, spans)
            vectors = np.zeros((2, 384), dtype=np.float32)
            vectors[0, 0] = vectors[1, 1] = 1
            index = faiss.IndexFlatIP(384)
            index.add(vectors)
            corpus = {'status': 'complete', 'corpus_hash': s.digest(text), 'spans': spans,
                      'documents': [{'kind': 'html', 'title': 'Synthetic unit fixture', 'pages': [{'text': text}]}],
                      'visited': [s.BASE_URL + 'unit-fixture'], 'failures': {}, 'base_url': s.BASE_URL,
                      'mode': 'full', 'created_at': s.utc_now(), 'html_parser': s.HTML_PARSER_VERSION}
            s.atomic_bytes(folder / 'RAGretrievalcontentfile.txt', text.encode())
            model_key = {'revision': None, 'synthetic_fixture': True}
            embedding_key = s.digest(s.embedding_configuration(chunks, model_key))
            manifest = s.publish_runtime_bundle(folder / 'runtime', corpus, chunks, metadata, index, embedding_key, 'unit-index', None)
            restored = s.restore_published_corpus(folder, folder / 'runtime', s.BASE_URL)
            self.assertEqual(restored['text'], text)
            c, m, idx, info = s.load_runtime_bundle(folder / 'runtime')
            self.assertEqual(c, chunks)
            self.assertEqual(idx.ntotal, 2)
            recovered = s.restore_published_embeddings(folder / 'runtime', chunks, model_key)
            self.assertIsNotNone(recovered)
            self.assertTrue(np.array_equal(recovered['matrix'], vectors))
            self.assertIsNone(s.restore_published_embeddings(folder / 'runtime', chunks, {'revision': 'different-fixture'}))
            s.atomic_bytes(folder / 'RAGretrievalcontentfile.txt', b'corrupt')
            with self.assertRaises(ValueError):
                s.restore_published_corpus(folder, folder / 'runtime', s.BASE_URL)
            (folder / 'runtime' / manifest['files']['index']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                s.load_runtime_bundle(folder / 'runtime')

    def test_generic_memory_evidence_does_not_hardcode_persona(self):
        import re
        helper_tree = ast.parse((ROOT / '_implementation/h1.py').read_text(encoding='utf8'))
        node = next(n for n in helper_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prior_personal_statements')
        namespace = {'re': re}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual conversation retrieval>', 'exec'), namespace)
        messages = [HumanMessage(content='My name is Bob and I am a student in EE.'), AIMessage(content='Hello'),
                    HumanMessage(content='What is my name and department?')]
        self.assertEqual(namespace['prior_personal_statements'](messages, messages[-1].content), [messages[0].content])
        self.assertEqual(namespace['prior_personal_statements']([messages[-1]], messages[-1].content), [])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests))
    s.atomic_json(ROOT / '.a6_artifacts/reports/unit_tests.json', {'tests_run': result.testsRun,
                  'failures': len(result.failures), 'errors': len(result.errors), 'successful': result.wasSuccessful(),
                  'fixture_only': True, 'at': s.utc_now()})
    raise SystemExit(0 if result.wasSuccessful() else 1)
