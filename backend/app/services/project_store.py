"""Project-scoped agency workflow. Decisions serialize with submission revisions."""
from __future__ import annotations

from contextlib import closing
from datetime import timedelta
from uuid import uuid4

from . import contributor_store as cs


JSON_FIELDS = ("product_codes", "company_ids", "approval_steps", "channels", "kb_products")


def unpack(row):
    item = dict(row)
    for key in JSON_FIELDS:
        item[key] = cs._loads(item.pop(key + "_json"), {} if key == "kb_products" else [])
    return item


def get_project(project_id, conn=None):
    if conn is None:
        with closing(cs._connect()) as db:
            return get_project(project_id, db)
    row = conn.execute("SELECT * FROM geo_projects WHERE project_id=?", (project_id,)).fetchone()
    if not row:
        raise ValueError("项目不存在")
    return unpack(row)


def can_access(project, user):
    return (user["role"] == "admin" or user["username"] == project["owner_username"]
            or any(s["username"] == user["username"] for s in project["approval_steps"]))


def require_access(project, user, *, manage=False):
    allowed = (user["role"] == "admin" or user["username"] == project["owner_username"]) if manage else can_access(project, user)
    if not allowed:
        raise PermissionError("无权访问该项目")


def active(project):
    if project["status"] != "active":
        raise ValueError("项目已归档，不能继续流转")


def event(conn, project_id, actor, action, details):
    conn.execute("INSERT INTO geo_project_events VALUES (?,?,?,?,?,?)",
                 (uuid4().hex, project_id, actor, action, cs._json(details), cs._iso()))


def create_project(values, user):
    name, brief = values["name"].strip(), values["brief"].strip()
    products = list(dict.fromkeys(values["product_codes"]))
    companies = list(dict.fromkeys(values["company_ids"]))
    steps = values["approval_steps"]
    if not name or not brief or not products or not companies or not steps:
        raise ValueError("请填写项目名称、Brief、产品、合作公司及审批节点")
    if not set(products) <= {p["product_code"] for p in cs.list_products()}:
        raise ValueError("包含无效产品")
    if any(not s["username"].strip() or not s["name"].strip() for s in steps):
        raise ValueError("审批节点须填写岗位名称和门户用户名")
    steps = [{"name": s["name"].strip(), "username": s["username"].strip()} for s in steps]
    kb_products = values.get("kb_products", {})
    if not set(kb_products) <= set(products) or any(not v.strip() for v in kb_products.values()):
        raise ValueError("知识库产品映射无效")
    for field in ("submission_deadline", "monitor_until"):
        if values.get(field):
            parsed = cs._parse_time(values[field])
            if parsed <= cs._now_dt():
                raise ValueError("截止时间必须晚于当前时间")
            from datetime import timezone
            values[field] = cs._iso(parsed.astimezone(timezone.utc))
    interval = values.get("monitor_interval_hours", 24)
    if not 1 <= interval <= 720:
        raise ValueError("监测间隔须为 1–720 小时")
    pid, now = "prj_" + uuid4().hex, cs._iso()
    with closing(cs._connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        for cid in companies:
            row = conn.execute("SELECT is_active FROM contributor_companies WHERE company_id=?", (cid,)).fetchone()
            if not row or not row[0]:
                raise ValueError("合作公司不存在或已停用")
        conn.execute("""INSERT INTO geo_projects
            (project_id,name,owner_username,brief,product_codes_json,company_ids_json,
             approval_steps_json,channels_json,target_questions,kb_products_json,
             submission_deadline,monitor_interval_hours,monitor_until,next_monitor_at,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pid, name, user["username"], brief, cs._json(products), cs._json(companies), cs._json(steps),
             cs._json(values.get("channels", [])), values.get("target_questions", ""), cs._json(kb_products),
             values.get("submission_deadline"), interval, values.get("monitor_until"), now, now, now))
        event(conn, pid, user["username"], "created", {"approval_steps": steps})
        conn.commit()
    return get_project(pid)


def list_projects(user):
    with closing(cs._connect()) as conn:
        rows = conn.execute("SELECT * FROM geo_projects ORDER BY created_at DESC").fetchall()
    return [p for p in map(unpack, rows) if can_access(p, user)]


def workspace_project(project_id):
    p = get_project(project_id)
    return {k: p[k] for k in ("project_id", "name", "brief", "channels", "target_questions", "submission_deadline", "status")}


def validate_invite(conn, invite, *, new_submission=False):
    """Recheck live authorization inside each mutation's transaction."""
    row = conn.execute("""SELECT i.*,c.is_active FROM contributor_invites i
        JOIN contributor_companies c USING(company_id) WHERE invite_id=?""", (invite["invite_id"],)).fetchone()
    if (not row or row["company_id"] != invite["company_id"] or row["revoked_at"] or not row["is_active"]
            or cs._parse_time(row["expires_at"]) <= cs._now_dt()):
        raise ValueError("邀请已失效")
    if row["project_id"] != invite.get("project_id"):
        raise ValueError("项目授权不匹配")
    if row["project_id"]:
        project = get_project(row["project_id"], conn)
        active(project)
        if row["company_id"] not in project["company_ids"]:
            raise ValueError("公司未获项目授权")
        if new_submission and project["submission_deadline"] and cs._parse_time(project["submission_deadline"]) <= cs._now_dt():
            raise ValueError("项目已截止投稿")
    return row


def check_submission_scope(row, session):
    if not row or row["company_id"] != session["company_id"] or row["project_id"] != session.get("project_id"):
        raise ValueError("投稿不存在")


def decide(submission_id, user, action, version, step, feedback, finding_ids):
    if action not in {"approve", "request_revision", "reject"}:
        raise ValueError("无效审批操作")
    with closing(cs._connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM article_submissions WHERE submission_id=?", (submission_id,)).fetchone()
        if not row or not row["project_id"]:
            raise ValueError("项目稿件不存在")
        p = get_project(row["project_id"], conn)
        require_access(p, user)
        active(p)
        if row["status"] != "awaiting_admin" or row["current_version"] != version or row["approval_step"] != step:
            raise ValueError("稿件版本或审批节点已变化，请刷新后重试")
        if step >= len(p["approval_steps"]) or p["approval_steps"][step]["username"] != user["username"]:
            raise PermissionError("仅当前指定审批人可以作出决定")
        if action != "approve" and not feedback.strip():
            raise ValueError("退回或不予采用须填写原因")
        report = conn.execute("""SELECT r.* FROM article_review_reports r JOIN article_review_jobs j USING(job_id)
            WHERE j.submission_id=? AND j.version=? AND j.status='success'""", (submission_id, version)).fetchone()
        if not report:
            raise ValueError("当前版本尚未完成知识核验")
        if action == "approve":
            unresolved = conn.execute("SELECT count(*) FROM article_review_findings WHERE job_id=? AND blocks_publication=1 AND coalesce(reviewer_note,'')=''", (report["job_id"],)).fetchone()[0]
            if unresolved:
                raise ValueError("仍有阻断问题，请退回修改或逐项核实并记录误报依据")
        final = action == "approve" and step == len(p["approval_steps"]) - 1
        state = "approved_waiting_publication" if final else "awaiting_admin" if action == "approve" else "revision_requested" if action == "request_revision" else "rejected"
        now = cs._iso()
        conn.execute("INSERT INTO article_approval_decisions VALUES (?,?,?,?,?,?,?,?)",
                     (uuid4().hex, submission_id, version, step, user["username"], action, feedback, now))
        conn.execute("""UPDATE article_submissions SET status=?,approval_step=?,admin_feedback=?,
            approved_version=?,approved_by=?,approved_at=?,updated_at=? WHERE submission_id=?""",
            (state, step + 1 if action == "approve" else step, feedback,
             version if final else None, user["username"] if final else None, now if final else None, now, submission_id))
        # Only the explicit reviewer selection is exposed to the contributor.
        conn.execute("UPDATE article_review_findings SET external_visible=0 WHERE job_id=?", (report["job_id"],))
        for fid in finding_ids:
            conn.execute("UPDATE article_review_findings SET external_visible=1 WHERE job_id=? AND finding_id=?", (report["job_id"], fid))
        cs._event(conn, submission_id, "reviewer", user["username"], action, row["status"], state,
                  {"version": version, "step": step, "feedback": feedback})
        conn.commit()
    return cs.get_submission(submission_id)


def resolve_finding(submission_id, finding_id, user, version, step, note):
    if not note.strip():
        raise ValueError("请填写误报核实依据")
    with closing(cs._connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM article_submissions WHERE submission_id=?", (submission_id,)).fetchone()
        if not row or not row["project_id"]:
            raise ValueError("稿件不存在")
        p = get_project(row["project_id"], conn)
        active(p)
        if row["status"] != "awaiting_admin" or row["current_version"] != version or row["approval_step"] != step:
            raise ValueError("稿件已变化，请刷新")
        if p["approval_steps"][step]["username"] != user["username"]:
            raise PermissionError("仅当前指定审批人可核实问题")
        cur = conn.execute("""UPDATE article_review_findings SET reviewer_note=? WHERE finding_id=?
            AND job_id=(SELECT job_id FROM article_review_jobs WHERE submission_id=? AND version=?)""",
            (note.strip(), finding_id, submission_id, version))
        if not cur.rowcount:
            raise ValueError("问题项不存在")
        cs._event(conn, submission_id, "reviewer", user["username"], "finding_resolved", row["status"], row["status"], {"finding_id": finding_id, "note": note, "version": version})
        conn.commit()


def accept_publication(submission_id, user, version, publication_revision, action, feedback):
    if action not in {"accept", "correct_metadata", "revise_content"} or not feedback.strip():
        raise ValueError("请选择验收结论并填写核对依据或整改意见")
    with closing(cs._connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM article_submissions WHERE submission_id=?", (submission_id,)).fetchone()
        if not row or not row["project_id"]:
            raise ValueError("项目稿件不存在")
        p = get_project(row["project_id"], conn)
        require_access(p, user, manage=True)
        active(p)
        if row["status"] != "publication_pending" or row["current_version"] != version or row["approved_version"] != version or row["publication_revision"] != publication_revision:
            raise ValueError("发布记录已变化，请刷新后重新验收")
        state = {"accept": "publication_accepted", "correct_metadata": "approved_waiting_publication", "revise_content": "revision_requested"}[action]
        conn.execute("UPDATE article_submissions SET status=?,admin_feedback=?,updated_at=? WHERE submission_id=?", (state, feedback, cs._iso(), submission_id))
        cs._event(conn, submission_id, "owner", user["username"], "publication_" + action, row["status"], state,
                  {"version": version, "publication_revision": publication_revision, "url": row["published_url"], "feedback": feedback})
        conn.commit()
    if action == "accept":
        cs.promote_submission(submission_id, p["owner_username"])
    return cs.get_submission(submission_id)


def finish_project(project_id, user, notes, archive=False):
    with closing(cs._connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        p = get_project(project_id, conn)
        require_access(p, user, manage=True)
        active(p)
        if not notes.strip():
            raise ValueError("请填写复盘记录")
        if archive and conn.execute("SELECT count(*) FROM article_submissions WHERE project_id=? AND status NOT IN ('tracked','rejected')", (project_id,)).fetchone()[0]:
            raise ValueError("仍有未完成稿件，请完成处理后归档")
        state = "archived" if archive else "active"
        conn.execute("UPDATE geo_projects SET review_notes=?,status=?,updated_at=? WHERE project_id=?", (notes, state, cs._iso(), project_id))
        event(conn, project_id, user["username"], "archived" if archive else "retrospective", {"notes": notes})
        conn.commit()
    return get_project(project_id)


def capture_metrics(project_id):
    """Observe the owner's imported GEO samples, not other users' private datasets."""
    from collections import defaultdict
    p = get_project(project_id)
    active(p)
    with closing(cs._connect()) as conn:
        articles = conn.execute("""SELECT s.submission_id,s.title,a.article_id,ap.url_match_key,ap.published_at
            FROM article_submissions s JOIN outbound_articles a ON a.article_id=s.article_id
            JOIN article_publications ap ON ap.article_id=a.article_id
            WHERE s.project_id=? AND s.status='tracked'""", (project_id,)).fetchall()
        products = p["product_codes"]
        marks = ','.join('?' for _ in products)
        samples = conn.execute(f"""SELECT a.dataset_id,a.answer_id,a.question_id,a.model,a.timestamp
            FROM answers a JOIN datasets d USING(dataset_id)
            WHERE d.owner_username=? AND a.product_code IN ({marks})""",
            (p["owner_username"], *products)).fetchall()
        sources = conn.execute(f"""SELECT s.dataset_id,s.answer_id,s.source_index,s.url
            FROM sources s JOIN datasets d USING(dataset_id)
            JOIN answers a ON a.dataset_id=s.dataset_id AND a.answer_id=s.answer_id
            WHERE d.owner_username=? AND a.product_code IN ({marks})""",
            (p["owner_username"], *products)).fetchall()
    sample_map = {(a['dataset_id'], a['answer_id']): a for a in samples}
    by_url = defaultdict(list)
    for a in articles:
        by_url[a['url_match_key']].append(a)
    matches, models, questions, cited = [], defaultdict(set), set(), set()
    for source in sources:
        answer = sample_map.get((source['dataset_id'], source['answer_id']))
        if not answer or not answer['timestamp']:
            continue
        try:
            _, key = cs.url_match_key(source['url'] or '')
            at = cs._parse_time(answer['timestamp'])
        except (ValueError, TypeError):
            continue
        for article in by_url.get(key, []):
            if not article['published_at'] or at < cs._parse_time(article['published_at']):
                continue
            cited.add(article['article_id'])
            questions.add((answer['dataset_id'], answer['question_id']))
            models[answer['model']].add((answer['dataset_id'], answer['answer_id']))
            matches.append({**dict(source), 'title': article['title'], 'model': answer['model'],
                            'question_id': answer['question_id'], 'article_id': article['article_id']})
    result = {'sample_answers': len(samples), 'dataset_count': len({a['dataset_id'] for a in samples}),
              'tracked_articles': len({a['article_id'] for a in articles}), 'cited_articles': len(cited),
              'citation_refs': len(matches), 'covered_questions': len(questions),
              'models': {key: len(value) for key, value in models.items()},
              'evidence': matches[:200], 'evidence_truncated': len(matches) > 200,
              'scope': '项目负责人已导入的相关产品 GEO 批次；累计精确 URL 引用；仅统计晚于实际发布时间的引用',
              'data_status': 'observed' if samples else 'no_samples'}
    now = cs._iso()
    with closing(cs._connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        active(get_project(project_id, conn))
        conn.execute("INSERT INTO geo_project_monitor_snapshots VALUES (?,?,?,?)", (uuid4().hex, project_id, now, cs._json(result)))
        conn.execute("UPDATE geo_projects SET last_monitor_at=?,monitor_error=NULL WHERE project_id=?", (now, project_id))
        conn.commit()
    return result


def monitor_tick():
    """Durable periodic observation and recovery after acceptance/promotion interruption."""
    with closing(cs._connect()) as conn:
        recover = conn.execute("SELECT submission_id,approved_by FROM article_submissions WHERE status='publication_accepted'").fetchall()
    for row in recover:
        try:
            cs.promote_submission(row['submission_id'], get_project(cs.get_submission(row['submission_id'])['project_id'])['owner_username'])
        except ValueError:
            pass
    with closing(cs._connect()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        now = cs._iso()
        rows = conn.execute("""SELECT * FROM geo_projects WHERE status='active' AND next_monitor_at<=?
            AND (monitor_until IS NULL OR monitor_until>?) LIMIT 10""", (now, now)).fetchall()
        for row in rows:
            conn.execute("UPDATE geo_projects SET next_monitor_at=? WHERE project_id=?",
                         (cs._iso(cs._now_dt() + timedelta(hours=row['monitor_interval_hours'])), row['project_id']))
        conn.commit()
    for row in rows:
        try:
            capture_metrics(row['project_id'])
        except Exception:
            with closing(cs._connect()) as conn:
                conn.execute("UPDATE geo_projects SET monitor_error=? WHERE project_id=?", ('本轮监测失败，可在项目工作台重试', row['project_id']))
                conn.commit()


def project_detail(project_id, user):
    p = get_project(project_id)
    require_access(p, user)
    p['can_manage'] = user['role'] == 'admin' or user['username'] == p['owner_username']
    p['submissions'] = cs.list_submissions(project_id=project_id)
    with closing(cs._connect()) as conn:
        p['events'] = [dict(r) for r in conn.execute('SELECT * FROM geo_project_events WHERE project_id=? ORDER BY created_at DESC LIMIT 100', (project_id,))]
        p['snapshots'] = [{**dict(r), 'result': cs._loads(r['result_json'], {})} for r in conn.execute(
            'SELECT captured_at,result_json FROM geo_project_monitor_snapshots WHERE project_id=? ORDER BY captured_at DESC LIMIT 30', (project_id,))]
        p['invites'] = [r for r in cs.list_invites() if r['project_id'] == project_id] if p['can_manage'] else []
    return p
