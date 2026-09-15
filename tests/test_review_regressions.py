from __future__ import annotations

import asyncio
import importlib.util
import json
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from backend.app.services import investigation_store, job_store, user_config_store
from utils.question_validation import validate_questions


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), ROOT / 'scripts' / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def job_env(tmp_path, monkeypatch):
    monkeypatch.setattr(job_store, 'JOBS_DB_PATH', tmp_path / 'jobs.sqlite')
    monkeypatch.setattr(job_store, 'JOBS_DIR', tmp_path / 'jobs')
    monkeypatch.setattr(job_store, 'ensure_worker', Mock())
    monkeypatch.setattr(job_store, '_QUEUE', Mock())
    monkeypatch.setattr(job_store, '_CANCEL_FLAGS', set())
    monkeypatch.setattr(job_store, '_stage_env', lambda *args: {})
    monkeypatch.setattr(job_store, 'load_models', lambda: {'models': {'qwen': {'enabled': True}}})
    monkeypatch.setattr(investigation_store, 'scan_dataset', lambda *args, **kwargs: {'scan_id': 'test', 'candidate_count': 0})
    job_store.init_db()


@pytest.mark.parametrize('question_id', ['../../escape', '/tmp/escape', r'C:\escape', '..\\escape', 'a/b', '', 'x' * 181])
def test_job_rejects_unsafe_ids_before_writing(job_env, question_id):
    with pytest.raises(ValueError, match='ID'):
        job_store.create_job('alice', 'user', 'test', [
            {'id': question_id, 'question': 'test', 'product': 'p1'},
        ], ['qwen'])
    assert job_store.list_jobs() == []
    assert not job_store.JOBS_DIR.exists()


def test_question_validation_rejects_duplicates_and_invalid_structure():
    question = {'id': 'p1_q4_0001', 'question': 'test', 'product': 'p1'}
    with pytest.raises(ValueError, match='重复'):
        validate_questions([question, question])
    with pytest.raises(ValueError, match='正文'):
        validate_questions([dict(question, question=' ')])
    validate_questions([question])


@pytest.mark.parametrize('failed_stage', ['collect', 'analyze', 'extract', 'verify', 'import', 'materialize'])
def test_pipeline_failure_stays_retryable(job_env, monkeypatch, failed_stage):
    item = job_store.create_job('alice', 'user', 'test', [
        {'id': 'q1', 'question': 'test', 'product': 'p1'},
    ], ['qwen'])
    stages = []

    def run_stage(name, *args):
        stages.append(name)
        return 1 if name == failed_stage else 0

    monkeypatch.setattr(job_store, '_run_stage', run_stage)
    job_store._run_job(item['job_id'])
    failed = job_store.get_job(item['job_id'])
    assert failed['status'] == 'failed'
    assert failed['stage'] == failed_stage
    assert stages[-1] == failed_stage
    assert failed['error']
    job_store.retry_job(item['job_id'])
    monkeypatch.setattr(job_store, '_run_stage', lambda *args: 0)
    job_store._run_job(item['job_id'])
    assert job_store.get_job(item['job_id'])['status'] == 'success'


def test_legacy_config_is_neither_read_nor_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(user_config_store, 'USER_CONFIGS_DIR', tmp_path / 'users')
    monkeypatch.setattr(user_config_store, 'GLOBAL_KB_PATH', tmp_path / 'default.json')
    legacy = tmp_path / 'users' / 'ab'
    legacy.mkdir(parents=True)
    (legacy / 'knowledge_base.json').write_text('{"private": "legacy"}')
    (legacy / 'brands.yaml').write_text('private: legacy')
    assert user_config_store.load_effective_kb('a.b')['data'] == {}
    assert user_config_store.user_brands_path('a.b') is None
    user_config_store.reset_user_kb('a.b')
    user_config_store.reset_user_brands('a.b')
    assert (legacy / 'knowledge_base.json').exists()
    assert (legacy / 'brands.yaml').exists()
    user_config_store.save_user_kb('a.b', {'own': 'alice'})
    user_config_store.save_user_kb('ab', {'own': 'bob'})
    user_config_store.reset_user_kb('a.b')
    assert user_config_store.load_effective_kb('ab')['data'] == {'own': 'bob'}
    # 全由符号组成的有效门户用户名也无需经过旧版清洗。
    assert user_config_store.load_effective_kb('...')['data'] == {}


@pytest.fixture
def collector_env(tmp_path, monkeypatch):
    module = script('03_query_models.py')
    config = tmp_path / 'config'
    config.mkdir()
    spec = {'enabled': True, 'supports_search': False, 'concurrency': 1, 'request_interval': 0}
    (config / 'models.yaml').write_text(json.dumps({'models': {'qwen': spec}, 'query_settings': {'retry_max': 2, 'retry_delay': 0}}))
    (config / 'api_keys.yaml').write_text('{"qwen":{"api_key":"test-only"}}')
    questions = tmp_path / 'questions.json'
    questions.write_text('[{"id":"q1","question":"test","product":"p1"}]')
    for key, value in {
        'BASE_DIR': str(tmp_path), 'RAW_DIR': str(tmp_path / 'raw'), 'LOG_PATH': str(tmp_path / 'execution.json'),
        'QUESTIONS_FILE_ENV': str(questions), 'ROUTE_ENV': 'direct', 'ROUNDS_ENV': '1',
        'MODELS_ENV': 'qwen', 'SEARCH_MODES_ENV': 'nosearch', 'MODEL_OVERRIDES_ENV': '',
    }.items():
        monkeypatch.setattr(module, key, value)
    client = SimpleNamespace(config=spec, supports_search=False, query=AsyncMock())
    monkeypatch.setattr(module, 'ModelClient', lambda *args, **kwargs: client)
    monkeypatch.setattr(module.AdaptiveThrottle, 'on_rate_limit', AsyncMock())
    return module, client


@pytest.mark.parametrize('message', ['HTTP 429 rate_limit', 'provider unavailable'])
def test_collector_failure_is_logged_and_resumed(collector_env, message):
    module, client = collector_env
    client.query.side_effect = RuntimeError(message)
    with pytest.raises(RuntimeError, match='采集未完成'):
        asyncio.run(module.main())
    log = module.load_execution_log()
    assert len(log['executions']) == 1
    assert log['executions'][0]['status'] == 'failed'
    assert client.query.call_count == 2
    client.query.side_effect = None
    client.query.return_value = {'answer': 'recovered', 'latency_ms': 1}
    asyncio.run(module.main())
    assert module.load_execution_log()['executions'][-1]['status'] == 'success'
    assert client.query.call_count == 3
    asyncio.run(module.main())
    assert client.query.call_count == 3


def test_collector_rejects_unsafe_direct_input(collector_env):
    module, client = collector_env
    Path(module.QUESTIONS_FILE_ENV).write_text('[{"id":"../../escape","question":"test","product":"p1"}]')
    with pytest.raises(ValueError, match='ID'):
        asyncio.run(module.main())
    client.query.assert_not_called()


def test_collector_repairs_missing_checkpoint_file(collector_env):
    module, client = collector_env
    client.query.return_value = {'answer': 'test', 'latency_ms': 1}
    asyncio.run(module.main())
    (Path(module.RAW_DIR) / 'qwen' / 'q1_r1_nosearch.json').unlink()
    asyncio.run(module.main())
    assert client.query.call_count == 2


def test_cancelled_collection_releases_capacity(collector_env):
    module, client = collector_env
    client.query.side_effect = asyncio.CancelledError()

    async def exercise():
        throttle = module.AdaptiveThrottle('qwen', 1, 0)
        with pytest.raises(asyncio.CancelledError):
            await module.execute_single_query(
                'qwen', client, {'id': 'q1', 'question': 'test', 'product': 'p1'},
                1, False, {}, {'executions': []}, set(), asyncio.Lock(), throttle,
                {'done': 0, 'total': 1},
            )
        assert throttle._active == 0
        await asyncio.wait_for(throttle.acquire(), 1)
        throttle.release()

    asyncio.run(exercise())


def test_extraction_failure_is_retried_but_successful_empty_is_cached():
    module = script('05_extract_recommendations.py')

    async def exercise():
        response = {'question_id': 'q1', 'answer': 'test'}
        client = AsyncMock()
        client.query.side_effect = RuntimeError('temporary failure')
        log, counter = {'completed': {}}, {'done': 0, 'skip': 0, 'fail': 0, 'total': 1}
        args = (client, response, asyncio.Semaphore(1), log, asyncio.Lock(), counter)
        await module.extract_one(*args)
        assert not log['completed']
        assert counter['fail'] == 1
        client.query.side_effect = None
        client.query.return_value = {'answer': '[]'}
        await module.extract_one(*args)
        assert module._make_extract_key(response) in log['completed']
        await module.extract_one(*args)
        assert client.query.call_count == 3
        assert counter['skip'] == 1

    asyncio.run(exercise())


def test_extraction_legacy_empty_cache_is_revalidated(tmp_path, monkeypatch):
    module = script('05_extract_recommendations.py')
    path = tmp_path / 'extraction_log.json'
    path.write_text('{"completed":{"failed_or_empty":[],"success":[{"product":"p1"}]}}')
    monkeypatch.setattr(module, 'EXTRACT_LOG_PATH', str(path))
    log = module.load_extract_log()
    assert 'failed_or_empty' not in log['completed']
    assert 'success' in log['completed']
    assert log['version'] == 2
    log['completed']['valid_empty'] = []
    module.save_extract_log(log)
    assert 'valid_empty' in module.load_extract_log()['completed']


@pytest.mark.parametrize('filename', ['questions_base.json', 'questions_expanded.json'])
def test_existing_question_files_remain_accepted(filename):
    validate_questions(json.loads((ROOT / 'questions' / filename).read_text(encoding='utf-8')))


def test_accuracy_provider_failure_cannot_finish_successfully(tmp_path, monkeypatch):
    module = script('07_verify_accuracy.py')
    config = tmp_path / 'config'
    config.mkdir()
    (config / 'models.yaml').write_text('{"models":{"deepseek":{}}}')
    (config / 'api_keys.yaml').write_text('{"deepseek":{"api_key":"test-only"}}')
    monkeypatch.setattr(module, 'BASE_DIR', str(tmp_path))
    monkeypatch.setattr(module, 'resolve_route', lambda *args: 'direct')
    monkeypatch.setattr(module, 'load_knowledge_base', lambda: {'p1': {'text': 'knowledge'}})
    monkeypatch.setattr(module, 'build_kb_resolver', lambda kb: lambda product: 'p1')
    monkeypatch.setattr(module, 'load_accuracy_responses', lambda: [{'question_id': 'p1_q1', 'product': 'p1', 'answer': 'test'}])
    client = AsyncMock()
    client.query.side_effect = RuntimeError('provider unavailable')
    monkeypatch.setattr(module, 'ModelClient', lambda *args, **kwargs: client)
    with pytest.raises(RuntimeError, match='准确率校验失败'):
        asyncio.run(module.run_verification())


def test_extraction_main_preserves_checkpoint_before_failing(tmp_path, monkeypatch):
    module = script('05_extract_recommendations.py')
    config = tmp_path / 'config'
    config.mkdir()
    (config / 'models.yaml').write_text('{"models":{"deepseek":{}}}')
    (config / 'api_keys.yaml').write_text('{"deepseek":{"api_key":"test-only"}}')
    monkeypatch.setattr(module, 'BASE_DIR', str(tmp_path))
    monkeypatch.setattr(module, 'EXTRACT_DIR', str(tmp_path / 'extractions'))
    monkeypatch.setattr(module, 'EXTRACT_LOG_PATH', str(tmp_path / 'extractions' / 'log.json'))
    monkeypatch.setattr(module, 'resolve_route', lambda *args: 'direct')
    monkeypatch.setattr(module, 'load_recommendation_responses', lambda: [{'question_id': 'q1', 'answer': 'test'}])
    client = AsyncMock()
    client.query.side_effect = RuntimeError('provider unavailable')
    monkeypatch.setattr(module, 'ModelClient', lambda *args, **kwargs: client)
    with pytest.raises(RuntimeError, match='推荐抽取失败'):
        asyncio.run(module.main())
    assert module.load_extract_log() == {'version': 2, 'completed': {}}


def test_nginx_deploy_replacement_is_idempotent_with_nested_public_location():
    deployment = (ROOT / 'scripts' / 'deploy_geo_subpath.sh').read_text(encoding='utf-8')
    embedded = deployment.split("ssh aliyun 'python3 - <<EOF\n", 1)[1].split("\nEOF'", 1)[0]
    config_path = '/etc/nginx/sites-enabled/aigc-creative-workflow'
    files = {
        config_path: 'server {\n    location /geo/ {\n        proxy_set_header X-Portal-Secret test-secret;\n    }\n}\n',
        '/tmp/nginx_geo_location.conf': (ROOT / 'scripts' / 'nginx_geo_location.conf').read_text(encoding='utf-8'),
    }

    class Writer:
        def __init__(self, path):
            self.path = path

        def write(self, text):
            files[self.path] = text

    def open_memory(path, mode='r'):
        return Writer(path) if mode == 'w' else io.StringIO(files[path])

    exec(compile(embedded, 'deploy_nginx_test', 'exec'), {'open': open_memory})
    first = files[config_path]
    exec(compile(embedded, 'deploy_nginx_test', 'exec'), {'open': open_memory})
    assert files[config_path] == first
    assert 'article-submit\\.html' in first
    assert first.count('auth_request off;') == 1


def test_rate_limit_keeps_queued_requests_and_respects_reduced_capacity():
    module = script('03_query_models.py')

    async def exercise():
        throttle = module.AdaptiveThrottle('test', 2, 0)
        await throttle.acquire()
        await throttle.acquire()
        finished = []

        async def queued(number):
            await throttle.acquire()
            assert throttle._active <= throttle.concurrency
            finished.append(number)
            await asyncio.sleep(0)
            throttle.release()

        tasks = [asyncio.create_task(queued(i)) for i in range(5)]
        await asyncio.sleep(0)
        pause = asyncio.create_task(throttle.on_rate_limit(0.01))
        await asyncio.sleep(0)
        throttle.release()
        await asyncio.sleep(0)
        assert not finished
        throttle.release()
        await asyncio.wait_for(asyncio.gather(pause, *tasks), 1)
        assert sorted(finished) == list(range(5))
        assert throttle._active == 0
        throttle._recover_threshold = 1
        throttle.on_success()
        assert throttle.concurrency == 2

    asyncio.run(exercise())


def test_analyze_single_response_generates_reports(tmp_path, monkeypatch):
    module = script('04_analyze_results.py')
    raw = tmp_path / 'raw'
    raw.mkdir()
    (raw / 'q1.json').write_text(json.dumps({
        'question_id': 'custom_q1', 'question_text': 'test', 'product': 'custom',
        'model': 'qwen', 'search_enabled': False, 'round': 1,
        'answer': 'No recommendation available.', 'sources': [],
    }), encoding='utf-8')
    analysis = tmp_path / 'analysis'
    monkeypatch.setattr(module, 'RAW_DIR', str(raw))
    monkeypatch.setattr(module, 'ANALYSIS_DIR', str(analysis))
    module.main()
    assert (analysis / 'raw_data.csv').exists()
    assert (analysis / 'optimization_report.csv').exists()
