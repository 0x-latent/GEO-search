from __future__ import annotations

import asyncio
from contextlib import closing
from datetime import timedelta
import hashlib
import json

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import pytest

from backend.app.services import contributor_store as cs, project_store as ps, article_review_service as review, knowledge_client as kb
from backend.app.api.project_routes import router as project_router
from backend.app.api.contributor_routes import router as contributor_router
import test_contributor_store as legacy_tests


OWNER = {'username': 'owner', 'role': 'user'}
EXPERT = {'username': 'expert', 'role': 'user'}
ADMIN = {'username': 'admin', 'role': 'admin'}


@pytest.fixture
def workflow():
    harness = legacy_tests.ContributorStoreTests()
    harness.setUp()
    company = cs.create_company('项目代理', 'admin')
    def create(name='项目 A'):
        p = ps.create_project({'name': name, 'brief': '核对产品表达', 'product_codes': ['p1'],
            'company_ids': [company['company_id']], 'approval_steps': [{'name': '专家', 'username': 'expert'}, {'name': '终审', 'username': 'owner'}],
            'channels': ['官网'], 'kb_products': {'p1': 'AN'}}, OWNER)
        inv = cs.create_invite(company['company_id'], 'owner', ['p1'], cs._iso(cs._now_dt()+timedelta(days=2)), 1, p['project_id'])
        token, _, _ = cs.exchange_invite(inv['invite_id'], inv['token'])
        session = cs.get_contributor_session(token)
        sub = cs.create_submission(session, 'article.txt', '产品说明正文'.encode(), 'p1', '稿件', '投稿人', 'a@example.com', planned_platform='官网')
        return p, session, sub, inv, token
    try:
        yield create
    finally:
        harness.tearDown()


def fixed(product, text):
    basis = {'role': 'agency', 'product': {'code': product, 'name': '测试产品'},
             'criteria': {'standard_expressions': [{'code': 'SE-AN-001', 'content': '产品说明正文'}], 'forbidden_expressions': [], 'compliance_notes': []},
             'standard_revisions': {'SE-AN-001': None}}
    return {**basis, 'format_version': 1, 'captured_at': cs._iso(), 'findings': [],
            'criteria_sha256': hashlib.sha256(kb.canonical(basis).encode()).hexdigest(),
            'input_text_sha256': hashlib.sha256(text.encode()).hexdigest()}


def finish_review(monkeypatch, findings=None):
    async def model(*args):
        return {'verdict': 'pass', 'risk_level': 'low', 'summary': '核验完成', 'findings': findings or []}, '{}', 'test', 'test', []
    monkeypatch.setattr(kb, 'fetch_context', fixed)
    monkeypatch.setattr(review, '_call_json', model)
    job = review.claim_jobs('worker', 1)[0]
    asyncio.run(review.process_job(job))


def approve_all(sid):
    ps.decide(sid, EXPERT, 'approve', 1, 0, '专家通过', [])
    return ps.decide(sid, OWNER, 'approve', 1, 1, '终审通过', [])


def test_full_workflow_requires_all_nodes_then_acceptance(workflow, monkeypatch):
    p, session, sub, _, _ = workflow()
    sid = sub['submission_id']
    with pytest.raises(ValueError):
        cs.update_publication(session, sid, '官网', 'https://example.com/a', expected_version=1)
    finish_review(monkeypatch)
    detail = cs.get_submission(sid)
    assert detail['status'] == 'awaiting_admin'
    assert json.loads(detail['report']['knowledge_snapshot_json'])['product']['code'] == 'AN'
    with pytest.raises(PermissionError):
        ps.decide(sid, ADMIN, 'approve', 1, 0, '', [])
    first = ps.decide(sid, EXPERT, 'approve', 1, 0, '通过', [])
    assert first['status'] == 'awaiting_admin' and first['approval_step'] == 1
    with pytest.raises(ValueError):
        ps.decide(sid, EXPERT, 'approve', 1, 0, '', [])
    approved = ps.decide(sid, OWNER, 'approve', 1, 1, '通过', [])
    assert approved['approved_version'] == 1 and approved['status'] == 'approved_waiting_publication'
    pub = cs.update_publication(session, sid, '官网', 'https://example.com/a', cs._iso(cs._now_dt()-timedelta(hours=1)), '核对发布页', 1)
    assert pub['status'] == 'publication_pending' and not pub['article_id']
    with pytest.raises(ValueError):
        cs.promote_submission(sid, 'owner')
    with pytest.raises(ValueError):
        ps.accept_publication(sid, OWNER, 1, 999, 'accept', '一致')
    accepted = ps.accept_publication(sid, OWNER, 1, pub['publication_revision'], 'accept', '已打开页面对照批准正文，一致')
    assert accepted['status'] == 'tracked' and accepted['article_id']
    assert ps.capture_metrics(p['project_id'])['data_status'] == 'no_samples'
    assert ps.finish_project(p['project_id'], OWNER, '已完成项目复盘', True)['status'] == 'archived'
    with pytest.raises(ValueError):
        ps.capture_metrics(p['project_id'])


def test_same_company_is_isolated_between_projects_and_legacy(workflow):
    p, session, sub, _, _ = workflow()
    p2, session2, sub2, _, _ = workflow('项目 B')
    assert len(cs.list_submissions(session['company_id'], external=True, project_id=p['project_id'])) == 1
    assert cs.list_submissions(session['company_id'], external=True) == []
    with pytest.raises(ValueError):
        cs.get_submission(sub['submission_id'], session2['company_id'], external=True, project_id=p2['project_id'])
    with pytest.raises(ValueError):
        cs.add_revision(session2, sub['submission_id'], 'new.txt', b'new')
    with pytest.raises(ValueError):
        cs.update_publication(session2, sub['submission_id'], '官网', 'https://example.com/a')
    assert not ps.list_projects({'username': 'stranger', 'role': 'user'})


def test_revision_resets_approval_and_cannot_reuse_stale_decision(workflow, monkeypatch):
    _, session, sub, _, _ = workflow()
    sid = sub['submission_id']
    finish_review(monkeypatch)
    approve_all(sid)
    new = cs.add_revision(session, sid, 'v2.txt', '新的产品正文'.encode())
    assert new['current_version'] == 2 and new['approval_step'] == 0 and not new['approved_version']
    assert len(cs.get_submission(sid)['decisions']) == 2
    finish_review(monkeypatch)
    with pytest.raises(ValueError):
        ps.decide(sid, EXPERT, 'approve', 1, 0, '', [])
    assert ps.decide(sid, EXPERT, 'approve', 2, 0, '', [])['status'] == 'awaiting_admin'


def test_blocking_finding_must_be_corrected_or_resolved(workflow, monkeypatch):
    _, _, sub, _, _ = workflow()
    sid = sub['submission_id']
    finish_review(monkeypatch, [{'excerpt': '原句', 'evidence': '判据', 'blocks_publication': True}])
    with pytest.raises(ValueError, match='阻断'):
        ps.decide(sid, EXPERT, 'approve', 1, 0, '直接通过', [])
    fid = cs.get_submission(sid)['findings'][0]['finding_id']
    with pytest.raises(PermissionError):
        ps.resolve_finding(sid, fid, OWNER, 1, 0, '误报')
    ps.resolve_finding(sid, fid, EXPERT, 1, 0, '已核实来源及原文，属于误报')
    assert ps.decide(sid, EXPERT, 'approve', 1, 0, '', [fid])['approval_step'] == 1


def test_quota_does_not_prevent_return_visit_and_revocation_stops_mutations(workflow):
    _, session, sub, inv, _ = workflow()
    assert cs.exchange_invite(inv['invite_id'], inv['token'])[1]['remaining_submissions'] == 0
    cs.revoke_invite(inv['invite_id'])
    with pytest.raises(ValueError, match='失效'):
        cs.add_revision(session, sub['submission_id'], 'v2.txt', b'body')


def test_missing_kb_stops_project_even_if_local_knowledge_exists(workflow, monkeypatch):
    _, _, sub, _, _ = workflow()
    monkeypatch.delenv('GEO_KB_URL', raising=False)
    monkeypatch.delenv('GEO_KB_KEY', raising=False)
    monkeypatch.setattr(review, '_product_kb', lambda _: (_ for _ in ()).throw(AssertionError('must not use local fallback')))
    job = review.claim_jobs('worker', 1)[0]
    asyncio.run(review.process_job(job))
    assert cs.get_submission(sub['submission_id'])['status'] == 'blocked_missing_kb'


def test_kb_contract_rejects_wrong_product_hash_and_future_snapshot():
    payload = fixed('AN', '产品说明正文')
    assert kb.validate(payload, 'AN', '产品说明正文') == payload
    with pytest.raises(kb.KnowledgeUnavailable):
        kb.validate(payload, 'OTHER', '产品说明正文')
    with pytest.raises(kb.KnowledgeUnavailable):
        kb.validate(payload, 'AN', '不同正文')
    payload['criteria']['standard_expressions'][0]['content'] = '篡改的知识'
    with pytest.raises(kb.KnowledgeUnavailable):
        kb.validate(payload, 'AN', '产品说明正文')
    payload = fixed('AN', '产品说明正文')
    payload['captured_at'] = cs._iso(cs._now_dt()+timedelta(days=2))
    with pytest.raises(kb.KnowledgeUnavailable):
        kb.validate(payload, 'AN', '产品说明正文')


def test_http_routes_enforce_project_and_reviewer_permissions(workflow, monkeypatch):
    p, _, sub, _, token = workflow()
    finish_review(monkeypatch)
    p2, _, _, _, token2 = workflow('项目 B')
    app = FastAPI()
    @app.middleware('http')
    async def identity(request: Request, call_next):
        request.state.user = {'username': request.headers.get('test-user', 'stranger'), 'role': 'user'}
        return await call_next(request)
    app.include_router(project_router)
    app.include_router(contributor_router)
    with TestClient(app) as client:
        assert client.get('/api/projects').json() == []
        assert client.get(f"/api/projects/{p['project_id']}").status_code == 403
        path = f"/api/projects/{p['project_id']}/submissions/{sub['submission_id']}"
        assert client.get(path, headers={'test-user':'expert'}).status_code == 200
        assert client.get(f"/api/projects/{p2['project_id']}/submissions/{sub['submission_id']}", headers={'test-user':'owner'}).status_code == 404
        assert client.post(path+'/decisions', headers={'test-user':'owner'}, json={'action':'approve','version':1,'step':0}).status_code == 403
        client.cookies.set(cs.CONTRIBUTOR_COOKIE, token2)
        assert client.get(f"/api/contributor/submissions/{sub['submission_id']}").status_code == 404
        assert client.get(f"/api/contributor/submissions/{sub['submission_id']}/file").status_code == 404


def test_monitor_scheduler_and_pending_archive(workflow):
    p, _, _, _, _ = workflow()
    with pytest.raises(ValueError, match='未完成稿件'):
        ps.finish_project(p['project_id'], OWNER, '结束', True)
    ps.monitor_tick()
    ps.monitor_tick()
    assert len(ps.project_detail(p['project_id'], OWNER)['snapshots']) == 1


def test_monitor_counts_only_owner_product_and_post_publication_samples(workflow, monkeypatch):
    p, session, sub, _, _ = workflow()
    finish_review(monkeypatch)
    sid = sub['submission_id']
    approve_all(sid)
    published = cs._now_dt() - timedelta(days=1)
    pub = cs.update_publication(session, sid, '官网', 'https://example.com/a', cs._iso(published), '凭证', 1)
    ps.accept_publication(sid, OWNER, 1, pub['publication_revision'], 'accept', '正文一致')
    with closing(cs._connect()) as conn:
        for ds, owner in [('mine','owner'),('private','stranger')]:
            conn.execute("INSERT INTO datasets(dataset_id,name,source_type,imported_at,owner_username) VALUES (?,?, 'test',?,?)", (ds, ds, cs._iso(), owner))
            for product in ['p1','p2']:
                conn.execute("INSERT INTO questions(dataset_id,question_id,product_code,question_text) VALUES (?,?,?,?)", (ds, product, product, '问题'))
                for aid, at in [('before', published-timedelta(hours=1)), ('after', published+timedelta(hours=1))]:
                    answer_id = product + aid
                    conn.execute("""INSERT INTO answers(dataset_id,answer_id,question_id,product_code,model,search_enabled,round,timestamp,answer_text)
                        VALUES (?,?,?,?, 'testmodel',1,?,?, '引用回答')""", (ds, answer_id, product, product, 1 if aid == 'before' else 2, cs._iso(at)))
                    conn.execute("INSERT INTO sources(dataset_id,answer_id,source_index,url) VALUES (?,?,0,?)", (ds, answer_id, 'https://example.com/a'))
        conn.commit()
    result = ps.capture_metrics(p['project_id'])
    assert result['sample_answers'] == 2
    assert result['citation_refs'] == 1 and result['covered_questions'] == 1
    assert result['models'] == {'testmodel':1}
    assert result['evidence'][0]['dataset_id'] == 'mine'


def test_concurrent_approval_only_advances_once(workflow, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    _, _, sub, _, _ = workflow()
    finish_review(monkeypatch)
    def attempt():
        try:
            ps.decide(sub['submission_id'], EXPERT, 'approve', 1, 0, '', [])
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: attempt(), range(2))).count(True) == 1
    assert cs.get_submission(sub['submission_id'])['approval_step'] == 1


def test_returned_publication_requires_fresh_acceptance(workflow, monkeypatch):
    _, session, sub, _, _ = workflow()
    finish_review(monkeypatch)
    sid = sub['submission_id']
    approve_all(sid)
    at = cs._iso(cs._now_dt()-timedelta(hours=1))
    pub = cs.update_publication(session, sid, '官网', 'https://example.com/a', at, '凭证', 1)
    returned = ps.accept_publication(sid, OWNER, 1, pub['publication_revision'], 'correct_metadata', '修正链接')
    assert returned['status'] == 'approved_waiting_publication'
    corrected = cs.update_publication(session, sid, '官网', 'https://example.com/b', at, '新凭证', 1)
    with pytest.raises(ValueError):
        ps.accept_publication(sid, OWNER, 1, pub['publication_revision'], 'accept', '旧页面验收')
    assert ps.accept_publication(sid, OWNER, 1, corrected['publication_revision'], 'accept', '新链接核对一致')['status'] == 'tracked'


def test_kb_http_uses_geo_credential_and_fixed_role(monkeypatch):
    monkeypatch.setenv('GEO_KB_URL', 'https://kb.example.test')
    monkeypatch.setenv('GEO_KB_KEY', 'test-only-credential')
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, _): yield json.dumps(fixed('AN','正文')).encode()
    def post(url, **kwargs):
        assert url == 'https://kb.example.test/api/v1/brand/review-context'
        assert kwargs['headers'] == {'X-KB-App':'geo','X-KB-Key':'test-only-credential','X-KB-Role':'agency'}
        assert kwargs['allow_redirects'] is False and kwargs['json'] == {'product':'AN','text':'正文'}
        return Response()
    monkeypatch.setattr(kb.requests, 'post', post)
    assert kb.fetch_context('AN','正文')['role'] == 'agency'


def test_schema_upgrade_preserves_existing_records(workflow):
    from utils.sqlite_schema import ensure_schema
    p, _, sub, _, _ = workflow()
    with closing(cs._connect()) as conn:
        ensure_schema(conn)
        ensure_schema(conn)
    assert cs.get_submission(sub['submission_id'])['project_id'] == p['project_id']


def test_kb_deterministic_forbidden_expression_blocks_even_if_model_passes(workflow, monkeypatch):
    _, _, sub, _, _ = workflow()
    original = fixed
    def forbidden(product, text):
        doc = original(product, text)
        doc['criteria']['forbidden_expressions'] = [{'code':'FE-AN-001','expression':'产品说明正文','reason':'测试禁用表达'}]
        return doc
    finish_review(monkeypatch)
    # Trigger a new submitted version to exercise the actual worker with the fixed rule.
    sid = sub['submission_id']
    ps.decide(sid, EXPERT, 'request_revision', 1, 0, '补充材料', [])
    with closing(cs._connect()) as conn:
        invite = dict(conn.execute('SELECT * FROM contributor_invites WHERE invite_id=?', (sub['invite_id'],)).fetchone())
    cs.add_revision(invite, sid, 'v2.txt', '产品说明正文'.encode())
    monkeypatch.setattr(kb, 'fetch_context', forbidden)
    asyncio.run(review.process_job(review.claim_jobs('worker',1)[0]))
    detail = cs.get_submission(sid)
    assert detail['findings'][0]['blocks_publication'] == 1
    with pytest.raises(ValueError, match='阻断'):
        ps.decide(sid, EXPERT, 'approve', 2, 0, '', [])


def validated_context(payload, product, text):
    basis = {key: payload[key] for key in ('role', 'product', 'criteria', 'standard_revisions')}
    payload['criteria_sha256'] = hashlib.sha256(kb.canonical(basis).encode()).hexdigest()
    return kb.validate(payload, product, text)


@pytest.mark.parametrize('local_match', [False, True])
def test_kb_mandatory_hits_block_approval_even_if_model_passes(workflow, monkeypatch, local_match):
    _, _, sub, _, _ = workflow()

    def context(product, text):
        payload = fixed(product, text)
        payload['criteria']['forbidden_expressions'] = [{
            'code': 'FE-AN-001', 'expression': text if local_match else '另一禁用表达',
            'match_variants': ['替代表达'],
        }]
        payload['findings'] = [{
            'rule_source': 'brand', 'rule_code': 'FE-AN-001', 'severity': 'must',
            'message': '知识库确认此处命中禁用规则', 'matched_text': text,
            'suggestion': '请删除该宣称',
        }]
        return validated_context(payload, product, text)

    async def model(*args):
        return {'verdict': 'pass', 'findings': []}, '{}', 'test', 'test', []

    monkeypatch.setattr(kb, 'fetch_context', context)
    monkeypatch.setattr(review, '_call_json', model)
    asyncio.run(review.process_job(review.claim_jobs('worker', 1)[0]))
    detail = cs.get_submission(sub['submission_id'])
    assert detail['status'] == 'awaiting_admin'
    assert len(detail['findings']) == 1  # KB and local hits for the same rule are merged.
    finding = detail['findings'][0]
    assert finding['blocks_publication'] == 1
    assert finding['evidence'] == '知识库确认此处命中禁用规则'
    assert finding['suggestion'] == '请删除该宣称'
    with pytest.raises(ValueError, match='阻断'):
        ps.decide(sub['submission_id'], EXPERT, 'approve', 1, 0, '', [])


@pytest.mark.parametrize('text,expression,variants,expected', [
    ('绝对安全', '绝对安全', ['百分百安全'], True),
    ('百分百安全', '绝对安全', ['百分百安全'], True),
    ('最佳', '最佳', [], True),
    ('普通说明', '禁用', ['', '  '], False),
])
def test_local_screen_checks_original_variants_and_short_expressions(text, expression, variants, expected):
    criteria = {'forbidden_expressions': [{
        'code': 'FE-AN-001', 'expression': expression, 'match_variants': variants,
    }]}
    assert bool(kb.screen(text, criteria)) is expected


@pytest.mark.parametrize('mode', ['success', 'fallback', 'rejected'])
def test_large_project_context_reaches_models_without_truncation(workflow, monkeypatch, mode):
    _, _, sub, _, _ = workflow()
    snapshots, prompts = [], []

    def context(product, text):
        payload = fixed(product, text)
        standards = [{'code': f'SE-AN-{i:03}', 'content': 'A' * 2000} for i in range(1, 101)]
        payload['criteria']['standard_expressions'] = standards
        payload['standard_revisions'] = {row['code']: None for row in standards}
        payload['criteria']['compliance_notes'] = [{'content': '末尾合规条款也必须参与审核'}]
        payload = validated_context(payload, product, text)
        assert 160_000 < len(kb.canonical(payload).encode()) < kb.MAX_CONTEXT_BYTES
        snapshots.append(payload)
        return payload

    async def model(model_key, model_id, prompt, *args):
        prompts.append(prompt)
        if mode == 'rejected' or (mode == 'fallback' and len(prompts) == 1):
            raise ValueError('模型上下文长度限制')
        return {'verdict': 'pass', 'findings': []}, '{}', 'test', 'test', []

    monkeypatch.setattr(kb, 'fetch_context', context)
    monkeypatch.setattr(review, '_call_json', model)
    job = review.claim_jobs('worker', 1)[0]
    job['settings']['fallback_model_key'] = 'fallback' if mode == 'fallback' else None
    asyncio.run(review.process_job(job))
    assert len(prompts) == (2 if mode == 'fallback' else 1)
    for prompt in prompts:
        knowledge_json = prompt.split('产品知识库：\n', 1)[1].split('\n\n待审文章标题：', 1)[0]
        assert json.loads(knowledge_json) == snapshots[0]
    detail = cs.get_submission(sub['submission_id'])
    if mode == 'rejected':
        assert detail['status'] == 'review_failed'
        assert not detail['report']
        with pytest.raises(ValueError):
            ps.decide(sub['submission_id'], EXPERT, 'approve', 1, 0, '', [])
    else:
        assert detail['status'] == 'awaiting_admin'
        assert json.loads(detail['report']['knowledge_snapshot_json']) == snapshots[0]
