from __future__ import annotations

import os
from typing import Literal

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from .auth_routes import _current_user
from ..services import project_store as ps, contributor_store as cs

router = APIRouter(prefix="/api/projects")


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class Step(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    username: str = Field(min_length=1, max_length=128)


class ProjectPayload(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    brief: str = Field(min_length=1, max_length=10000)
    product_codes: list[str] = Field(min_length=1, max_length=100)
    company_ids: list[str] = Field(min_length=1, max_length=100)
    approval_steps: list[Step] = Field(min_length=1, max_length=10)
    channels: list[str] = Field(default_factory=list, max_length=50)
    kb_products: dict[str, str] = Field(default_factory=dict)
    target_questions: str = Field(default="", max_length=10000)
    submission_deadline: str | None = None
    monitor_interval_hours: int = Field(default=24, ge=1, le=720)
    monitor_until: str | None = None


class InvitePayload(BaseModel):
    company_id: str
    expires_at: str
    max_submissions: int = Field(default=20, ge=1, le=10000)


class DecisionPayload(BaseModel):
    action: Literal['approve', 'request_revision', 'reject']
    version: int = Field(ge=1)
    step: int = Field(ge=0)
    feedback: str = Field(default='', max_length=10000)
    finding_ids: list[str] = Field(default_factory=list)


class FindingPayload(BaseModel):
    version: int = Field(ge=1)
    step: int = Field(ge=0)
    note: str = Field(min_length=1, max_length=5000)


class AcceptancePayload(BaseModel):
    version: int = Field(ge=1)
    publication_revision: int = Field(ge=1)
    action: Literal['accept', 'correct_metadata', 'revise_content']
    feedback: str = Field(min_length=1, max_length=10000)


class FinishPayload(BaseModel):
    notes: str = Field(min_length=1, max_length=20000)
    archive: bool = False


def project_user(pid, request, manage=False):
    user = _current_user(request)
    p = call(ps.get_project, pid)
    call(ps.require_access, p, user, manage=manage)
    return p, user


def submission_user(pid, sid, request, manage=False):
    p, user = project_user(pid, request, manage)
    s = call(cs.get_submission, sid)
    if s['project_id'] != pid:
        raise HTTPException(404, '项目稿件不存在')
    return p, user, s


@router.get('/options')
def options(request: Request):
    _current_user(request)
    return {'products': cs.list_products(), 'companies': [{k: c[k] for k in ('company_id', 'name')} for c in cs.list_companies() if c['is_active']],
            'knowledge_connected': bool(os.environ.get('GEO_KB_URL') and os.environ.get('GEO_KB_KEY'))}


@router.get('')
def projects(request: Request):
    return ps.list_projects(_current_user(request))


@router.post('')
def create(payload: ProjectPayload, request: Request):
    return call(ps.create_project, payload.model_dump(), _current_user(request))


@router.get('/{pid}')
def detail(pid: str, request: Request):
    return call(ps.project_detail, pid, _current_user(request))


@router.post('/{pid}/invites')
def invite(pid: str, payload: InvitePayload, request: Request):
    p, user = project_user(pid, request, True)
    return call(cs.create_invite, payload.company_id, user['username'], p['product_codes'], payload.expires_at, payload.max_submissions, pid)


@router.delete('/{pid}/invites/{iid}')
def revoke(pid: str, iid: str, request: Request):
    project_user(pid, request, True)
    if not any(i['invite_id'] == iid and i['project_id'] == pid for i in cs.list_invites()):
        raise HTTPException(404, '邀请不存在')
    call(cs.revoke_invite, iid)
    return {'status': 'ok'}


@router.get('/{pid}/submissions/{sid}')
def submission(pid: str, sid: str, request: Request):
    return submission_user(pid, sid, request)[2]


@router.post('/{pid}/submissions/{sid}/decisions')
def decision(pid: str, sid: str, payload: DecisionPayload, request: Request):
    _, user, _ = submission_user(pid, sid, request)
    return call(ps.decide, sid, user, payload.action, payload.version, payload.step, payload.feedback, payload.finding_ids)


@router.post('/{pid}/submissions/{sid}/findings/{fid}')
def finding(pid: str, sid: str, fid: str, payload: FindingPayload, request: Request):
    _, user, _ = submission_user(pid, sid, request)
    call(ps.resolve_finding, sid, fid, user, payload.version, payload.step, payload.note)
    return cs.get_submission(sid)


@router.post('/{pid}/submissions/{sid}/acceptance')
def acceptance(pid: str, sid: str, payload: AcceptancePayload, request: Request):
    _, user, _ = submission_user(pid, sid, request, True)
    return call(ps.accept_publication, sid, user, payload.version, payload.publication_revision, payload.action, payload.feedback)


@router.post('/{pid}/submissions/{sid}/retry')
def retry(pid: str, sid: str, request: Request):
    p, user, _ = submission_user(pid, sid, request, True)
    call(ps.active, p)
    call(cs.retry_review, sid, user['username'])
    return {'status': 'ok'}


@router.post('/{pid}/monitor')
def monitor(pid: str, request: Request):
    project_user(pid, request, True)
    return call(ps.capture_metrics, pid)


@router.post('/{pid}/retrospective')
def finish(pid: str, payload: FinishPayload, request: Request):
    return call(ps.finish_project, pid, _current_user(request), payload.notes, payload.archive)
