"""固定判据 v1 契约：仅 agency 可见字段，不接受隐式类型转换。"""
from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_TEXT_CHARS = 20_000
MAX_CONTEXT_BYTES = 512 * 1024
Digest = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
ProductCode = Annotated[str, Field(pattern=r'^[A-Z0-9]{1,16}$')]
SECode = Annotated[str, Field(pattern=r'^SE-[A-Z0-9]{1,16}-[0-9]{1,4}$')]
FECode = Annotated[str, Field(pattern=r'^FE-[A-Z0-9]{1,16}-[0-9]{1,4}$')]
Nonblank = Annotated[str, Field(min_length=1)]


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class ReviewContextInput(Strict):
    product: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator('product', 'text')
    @classmethod
    def nonblank(cls, value):
        if not value.strip(): raise ValueError('产品及正文不能为空')
        return value  # Text must remain byte-for-byte identical to the submitted input.


class Product(Strict):
    code: ProductCode
    name: Nonblank


class Standard(Strict):
    code: SECode
    content: Nonblank
    expr_type: str | None = None
    scene: str | None = None
    evidence_level: str | None = None
    source_ref: str | None = None

    @field_validator('content')
    @classmethod
    def content_nonblank(cls, value):
        if not value.strip(): raise ValueError('标准表达内容不能为空')
        return value


class Forbidden(Strict):
    code: FECode
    expression: Nonblank
    forbid_type: str | None = None
    reason: str | None = None
    alternative_codes: list[SECode] = Field(default_factory=list)
    match_variants: list[str] = Field(default_factory=list)


class Compliance(Strict):
    code: str | None = None  # Historical CR may have no registered code.
    section: str | None = None
    section_kind: str | None = None
    content: Nonblank


class Criteria(Strict):
    standard_expressions: list[Standard]
    forbidden_expressions: list[Forbidden]
    compliance_notes: list[Compliance]


class Revision(Strict):
    revision_id: str
    revision_no: int = Field(ge=1)
    content_sha256: Digest

    @field_validator('revision_id')
    @classmethod
    def uuid_string(cls, value):
        if str(UUID(value)) != value: raise ValueError('版本标识无效')
        return value


class Finding(Strict):
    rule_source: Literal['brand']
    rule_code: FECode
    severity: Literal['must']
    message: str
    suggestion: str | None = None
    ref_code: SECode | None = None
    matched_text: str | None = None
    location: dict = Field(default_factory=dict)


class ReviewContext(Strict):
    format_version: Literal[1] = 1
    role: Literal['agency'] = 'agency'
    product: Product
    captured_at: str
    criteria: Criteria
    standard_revisions: dict[SECode, Revision | None]
    criteria_sha256: Digest
    input_text_sha256: Digest
    findings: list[Finding]

    @model_validator(mode='after')
    def consistent_product(self):
        standards=self.criteria.standard_expressions
        codes={r.code for r in standards}
        if len(codes)!=len(standards) or codes!=set(self.standard_revisions):
            raise ValueError('标准表达与版本映射不一致')
        seen=set(codes)
        for row in [*standards,*self.criteria.forbidden_expressions,*self.criteria.compliance_notes]:
            if row.code and row.code.split('-')[1:2]!=[self.product.code]:
                raise ValueError('条目跨产品')
        for row in [*self.criteria.forbidden_expressions,*self.criteria.compliance_notes]:
            if row.code and row.code in seen: raise ValueError('条目编号重复')
            if row.code: seen.add(row.code)
        for row in self.criteria.forbidden_expressions:
            if not row.expression.strip() or not set(row.alternative_codes)<=codes:
                raise ValueError('禁用表达或替代编号无效')
        if any(not r.content.strip() for r in self.criteria.compliance_notes):
            raise ValueError('合规提示为空')
        return self
