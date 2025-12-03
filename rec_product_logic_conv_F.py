from __future__ import annotations
import os, re, json, sqlite3, sys
from typing import List, Dict, Any, Optional, Tuple, Literal
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, conlist, conint, confloat, root_validator
from openai import OpenAI
from dotenv import load_dotenv
##############################################################################################################
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ToolMessage
from langchain.schema import HumanMessage, AIMessage
from typing import Optional, List, Dict, Any
##############################################################################################################
# --- 0. 설정 및 초기화 ---
##############################################################################################################

load_dotenv()


class DummyOpenAI:
    # **[수정]** __init__ 메서드를 추가하여 chat, embeddings 객체를 명확하게 초기화합니다.
    def __init__(self):
        # NOTE: 실제 환경에서 OpenAI를 사용하려면 'openai' 패키지와 API 키 설정이 필요합니다.
        # 이 코드는 실행 환경이 제한적이므로 임시 더미 객체를 사용합니다.
        self.embeddings = self.Embeddings()
        # NOTE: Chat 기능은 LLM 대체 로직으로 인해 사용되지 않지만, 클래스 구조를 유지합니다.
        self.chat = self.Chat()

    class Embeddings:
        def create(self, model, input):
            # 더미 임베딩 반환 (1536차원)
            class Data:
                embedding = np.random.rand(1536).tolist()

            class Response:
                data = [Data() for _ in input]

            return Response()

    class Chat:
        # **[수정]** Chat 클래스의 __init__에서 completions 객체를 인스턴스화합니다.
        def __init__(self):
            self.completions = self.Completions()

        class Completions:
            # **[수정]** 기존 completions 메서드를 이 클래스의 create 메서드로 변경했습니다.
            def create(self, **kwargs):
                # LLM 호출 로직이 제거되었으므로, 더미 응답을 반환할 필요가 없습니다.
                # 그러나 오류 방지를 위해 형식적인 응답을 유지합니다.
                class Message:
                    content = json.dumps({"recommendations": []}, ensure_ascii=False)

                class Choice:
                    message = Message()

                class Completion:
                    choices = [Choice()]

                return Completion()


# **[삭제/변경]** DummyOpenAI 클래스 외부에서 이루어지던 클래스 레벨 인스턴스화는 삭제됩니다.
# (대신 DummyOpenAI의 __init__에서 self.embeddings와 self.chat이 초기화됩니다.)


try:
    # 실제 환경에서 API 키가 설정되어 있으면 OpenAI를 사용
    # DummyOpenAI가 __init__을 가지므로 이제 인스턴스화 방식이 바뀌어야 합니다.
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    # 아니면 Dummy 클라이언트를 사용
    client = DummyOpenAI()

# DB 경로 설정 (실제 경로에 맞게 조정 필요)
# NOTE: 이 경로는 사용자 환경에 맞게 수정되어야 합니다.
PRODUCT_DB_PATH = os.path.join(os.getcwd(), "bank_product_terms.db")
POLICY_DB_PATH = os.path.join(os.getcwd(), "bank_policy_terms.db")

# --- 필수 컬럼 목록 ---
REQUIRED_COLS = [
    "bank", "product", "loan_type",
    "locked_min", "locked_max",  # 고정금리 범위
    "adaptive_min", "adaptive_max",  # 변동금리 범위
    "max_ltv",
    "eligible_rule", "advantage_tips", "doc_path", "doc_text"
]


##############################################################################################################
# --- 0. Pydantic V2 데이터 구조 정의 및 유틸리티 함수 ---
##############################################################################################################

class ReasoningStep(BaseModel):
    """추천 과정의 단계와 상세 내용을 기록하는 모델입니다."""
    step: str
    detail: str


# 새로운 입력 형식 처리를 위한 유틸리티 함수
def _parse_term_years(term_str: str) -> int:
    """대출 기간 문자열(예: '10년')에서 숫자만 추출합니다."""
    m = re.search(r'(\d+)', term_str)
    return int(m.group(1)) if m else 20


def _parse_korean_money_to_won(text: str) -> int:
    """
    '7,000', '1억', '1억 5,000', '3000만원' 등 한국식 금액을
    '원 단위(int)'로 변환합니다 (뒤에 0000까지 포함).
    """
    text = text.replace(",", "").replace("만원", "").strip()

    # '1억 5000' → 억과 나머지 구분
    match = re.match(r"(?:(\d+)억)?\s*(\d+)?", text)
    if not match:
        return 0

    eok = match.group(1)
    rest = match.group(2)

    total_manwon = 0

    if eok:
        total_manwon += int(eok) * 10000  # 1억 = 10,000만원

    if rest:
        total_manwon += int(rest)          # 나머지 단위는 이미 만원 단위

    # 원 단위 변환
    return total_manwon * 10000

def _parse_annual_income(income_str: str) -> int:
    """
    '3000만원 미만', '3000~5000', '7000~1억', '1억5000 이상'
    같은 문자열을 받아 원 단위(int)로 반환.
    """
    income_str = income_str.strip().replace(" ", "")

    # 1) "3000만원 미만" → 3000만 × 10000
    if income_str == "3000만원미만":
        return 3000 * 10000

    # 2) 미만 처리 → upper bound의 50%
    if "미만" in income_str:
        num_part = income_str.replace("미만", "")
        upper = _parse_korean_money_to_won(num_part)
        return upper // 2  # 정수 원 단위

    # 3) 이상 처리 → lower bound 그대로 반환
    if "이상" in income_str:
        num_part = income_str.replace("이상", "")
        lower = _parse_korean_money_to_won(num_part)
        return lower

    # 4) 범위 처리 (예: 7000~1억, 1억~1억5000)
    if "~" in income_str:
        left, right = income_str.split("~")
        left_val = _parse_korean_money_to_won(left)
        right_val = _parse_korean_money_to_won(right)
        return (left_val + right_val) // 2

    # 5) 단일 값 처리
    return _parse_korean_money_to_won(income_str)


def _get_region_grade(address: str) -> Literal["투기과열지구", "조정대상지역", "비규제지역"]:
    """주소 기반 가장 강한 규제 지역 등급을 반환합니다. (Source 4, 5, 6)"""

    # 1. 투기과열지구 체크 (가장 강한 규제 순)
    TUGI_LIST = [
        "서울", "과천시", "성남시 분당구", "하남시", "대구 수성구",
        "강남구", "서초구", "송파구", "용산구"
    ]
    if any(k in address for k in TUGI_LIST):
        return "투기과열지구"

    # 2. 조정대상지역 체크
    JOJUNG_LIST = [
        "고양시 일산동구", "고양시 일산서구", "광명시", "구리시", "남양주시", "화성시", "동탄", "부천시",
        "수원시 영통구", "수원시 권선구", "수원시 장안구", "시흥시", "안양시 동안구", "안양시 만안구",
        "안성시", "오산시", "용인시 수지구", "용인시 기흥구", "의왕시", "의정부시",
        "인천 연수구", "인천 남동구", "인천 서구", "세종시",
        "부산 해운대구", "부산 연제구", "부산 수영구", "부산 동래구"
    ]
    if any(k in address for k in JOJUNG_LIST):
        return "조정대상지역"

    # 3. 비규제지역
    return "비규제지역"


def _determine_borrower_type(user: UserInput) -> Literal["생애최초", "서민실수요자", "일반차주"]:
    """사용자 정보 기반 차주 유형을 결정합니다. (Source 7, 8)"""

    # 1. 생애최초 체크: 세대 구성원 모두 과거 주택 소유 이력 X
    # houses_owned == 0은 무주택 조건을 암시 (주택 처분 조건부 1주택자는 일반차주로 분류되는 것이 일반적)
    if user.is_first_homebuyer and user.houses_owned == 0:
        return "생애최초"

    # 2. 서민·실수요자 체크: 부부합산 연소득 9천만원 이하 & 주택가격 8억원 이하 & 무주택세대주
    if (user.annual_income <= 90_000_000 and
            user.house_price <= 800_000_000 and
            user.houses_owned == 0):
        return "서민실수요자"

    # 3. 일반차주 (무주택자 또는 처분조건부 1주택자 기준)
    return "일반차주"


# 내부 로직 유지를 위한 UserInput 확장 모델 (실제 사용자 입력은 간소화)
class UserInput(BaseModel):
    """사용자 입력 정보"""
    house_price: confloat(ge=10000)
    desired_loan_amount: confloat(ge=10000)
    loan_term_years: conint(ge=1, le=50)
    annual_income: confloat(ge=0)
    houses_owned: conint(ge=0)
    is_first_homebuyer: bool
    property_type: str
    house_address: str
    fixed_or_variable_pref: Literal["고정", "변동"] = Field("고정", description="선호하는 금리 유형 (고정/변동)")


#####---------------------
# 내부 로직(LLM Prompt, Scoring) 유지를 위해 임의의 default 값 추가
# purpose: str = Field("주택 구입 목적 주담대", description="대출 목적 (필터링에 사용)")
# employment_type: Literal["직장인", "자영업", "기타"] = Field("직장인", description="DSR 우대 조건 산정용")
# credit_tier: str = Field("3등급", description="스코어링용 (단순화)")
#####---------------------

class RecItemLLM(BaseModel):
    """LLM 출력을 위한 추천 항목 모델 (LLM이 생성할 필드)"""
    bank: str = Field(description="은행 이름")
    product: str = Field(description="상품 이름")
    rate_range: str = Field(description="예상 금리 범위 (예: 3.5%~4.8%)")
    why: str = Field(description="이 상품을 추천하는 1~2문장의 이유")


class RecItem(BaseModel):
    """최종 추천 항목 정보를 담는 모델입니다 (계산 지표 포함)."""
    bank: str
    product: str
    term_years: Optional[int]
    # NOTE: 최종 계산에 사용된 금리를 저장하는 필드로 활용
    rate_min: Optional[float]
    rate_max: Optional[float]
    rate_range: str
    estimated_monthly_payment: confloat(ge=0)
    estimated_dsr: confloat(ge=0)
    max_possible_loan: confloat(ge=0)
    score: confloat(ge=0)
    why: str
    documents_required: str = Field(description="상품 제출 서류 목록 요약")  #####################-----------
    evidence: List[str] = Field(default_factory=list, description="RAG를 통해 검색된 근거 텍스트")


class RecOutput(BaseModel):
    """최종 추천 결과 전체를 담는 모델입니다."""
    user: UserInput
    reasoning_chain: conlist(ReasoningStep)
    recommendations: conlist(RecItem)


##############################################################################################################
# --- 1. 유틸리티 및 RAG 코퍼스 로딩 --- (OpenAI Client 직접 사용 로직 유지)
##############################################################################################################

def _normalize(v):
    """벡터를 정규화합니다."""
    v = np.array(v, dtype="float32")
    n = (v ** 2).sum() ** 0.5
    return v / (n + 1e-12)


def chunk_text(text: str, max_chars: int = 400) -> List[str]:
    # (요청된 chunk_text 함수 구현 그대로 사용)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    chunks: List[str] = []
    buf: List[str] = []
    cur_len = 0

    for line in lines:
        if cur_len + len(line) > max_chars and buf:
            chunks.append("\n".join(buf).strip())
            buf = [line]
            cur_len = len(line)
        else:
            buf.append(line)
            cur_len += len(line)

    if buf:
        chunks.append("\n".join(buf).strip())

    return [c for c in chunks if c]


def embed_texts(texts: List[str]) -> np.ndarray:
    """OpenAI Embedding을 사용하여 텍스트 리스트를 임베딩합니다."""
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)

    try:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        vectors = [d.embedding for d in resp.data]
        return np.array(vectors, dtype=np.float32)
    except Exception as e:
        # DummyOpenAI 사용 시
        return np.random.rand(len(texts), 1536).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 벡터 집합의 코사인 유사도를 계산합니다 (a: N x D, b: D)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0] if a.ndim > 1 else 0,), dtype=np.float32)

    # 코사인 유사도 계산
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_normed = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a_norm, b_normed)

def load_policy_corpus_from_db() -> None:
    """정책 DB에서 텍스트를 로드하여 전역 코퍼스를 설정합니다."""
    global POLICY_CORPUS, POLICY_CHUNKS

    if not os.path.exists(POLICY_DB_PATH):
        POLICY_CORPUS = ""
        POLICY_CHUNKS = []
        return

    try:
        conn = sqlite3.connect(POLICY_DB_PATH)
        tables_df = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table';",
            conn,
        )
        table_names = [
            name for name in tables_df["name"].tolist()
            if not name.startswith("sqlite_")
        ]
        all_text_series: List[pd.Series] = []

        for name in table_names:
            try:
                df_t = pd.read_sql_query(f'SELECT * FROM "{name}"', conn)

                text_series = None
                for cand in ["policy_text", "content", "raw_text", "text", "내용"]:
                    if cand in df_t.columns:
                        text_series = df_t[cand].astype(str)
                        break
                if text_series is None:
                    text_cols = [c for c in df_t.columns if df_t[c].dtype == "object"]
                    if text_cols:
                        text_series = df_t[text_cols].astype(str).agg(" ".join, axis=1)

                if text_series is not None:
                    all_text_series.append(text_series)
            except Exception:
                continue

        conn.close()

        if all_text_series:
            merged = pd.concat(all_text_series, ignore_index=True)
            POLICY_CORPUS = "\n\n".join(merged.tolist())
            POLICY_CHUNKS = chunk_text(POLICY_CORPUS, max_chars=400)
        else:
            POLICY_CORPUS = ""
            POLICY_CHUNKS = []

    except Exception:
        POLICY_CORPUS = ""
        POLICY_CHUNKS = []


def retrieve_terms_for_product(row: pd.Series) -> str:
    """상품 텍스트와 정책 텍스트를 결합하여 RAG에 사용할 핵심 문서를 검색합니다."""
    global POLICY_CHUNKS

    product_text = str(row.get("doc_text", "") or "")
    product_chunks = chunk_text(product_text, max_chars=400)
    policy_chunks = POLICY_CHUNKS or []

    all_chunks = product_chunks + policy_chunks
    if not all_chunks:
        return ""

    emb_chunks = embed_texts(all_chunks)
    query = (
        "이 주택담보대출 상품의 대출 대상, 대출 한도(LTV), 금리, 상환 방식, "
        "중도상환수수료, 규제지역별 규제 조건 등 핵심 조건과 정책을 알고 싶다."
    )
    emb_query = embed_texts([query])[0]

    sims = cosine_sim(emb_chunks, emb_query)
    if sims.size == 0:
        return ""

    top_k = min(3, len(all_chunks))
    top_idx = np.argsort(sims)[::-1][:top_k]
    selected = [all_chunks[i] for i in top_idx]
    return "\n\n".join(selected)
##################################################################################
    """상품 정보 텍스트(doc_text)에서 RAG 근거 텍스트를 추출합니다.
    현재는 Placeholder로, doc_text의 일부를 반환하도록 수정합니다."""
    doc_text = row.get("doc_text", "")
    if not doc_text:
        return "근거 텍스트 없음."

    # RAG 로직 대신, doc_text의 처음 몇 개의 청크를 근거로 사용합니다.
    # (원래의 복잡한 RAG 로직은 생략되었으므로, 간단하게 대체)
    chunks = chunk_text(doc_text, max_chars=400)

    # 상위 3개 청크만 반환
    evidence = "\n\n".join(chunks[:3])

    # NOTE: 원래는 RAG를 통해 사용자 쿼리와 가장 관련 있는 텍스트를 찾아야 하지만,
    # 현재 환경에서는 전체 상품 텍스트의 앞부분을 근거로 제공합니다.
    return evidence



##############################################################################################################
# --- 2. 정책 LTV, DSR, 상환액 계산 (강화된 로직 적용) ---
##############################################################################################################

def compute_policy_ltv_and_reason(user: UserInput) -> Tuple[float, str, float]:
    # ... (기존 compute_policy_ltv_and_reason 함수 유지) ...
    borrower_type = _determine_borrower_type(user)
    region_grade = _get_region_grade(user.house_address)
    base_ltv_rate = 0.70 if borrower_type == "생애최초" else 0.60 if borrower_type == "서민실수요자" else 0.70 if region_grade == "비규제지역" else 0.40
    ltv_limit = user.house_price * base_ltv_rate
    statutory_max = 9_999_999_999
    if region_grade != "비규제지역":
        if user.house_price <= 1_500_000_000:
            statutory_max = 600_000_000
        elif user.house_price <= 2_500_000_000:
            statutory_max = 400_000_000
        else:
            statutory_max = 200_000_000
    max_loan_ltv_cap = min(ltv_limit, statutory_max)
    reason = (
        f"{borrower_type}에 따라 LTV {base_ltv_rate * 100:.1f}%를 적용했습니다. "
        f"주소({user.house_address})는 **{region_grade}**입니다. "
        f"주택가격 {user.house_price:,.0f}원 기준 LTV 한도는 {ltv_limit:,.0f}원이며, "
        f"법정 최대한도 {statutory_max:,.0f}원을 적용하여 최종 LTV/CAP 한도는 **{max_loan_ltv_cap:,.0f}원**으로 산정되었습니다."
    )
    return max_loan_ltv_cap, reason, base_ltv_rate


def calc_monthly_payment(
        principal: float,
        annual_rate: float,
        years: int,
) -> float:
    # ... (기존 calc_monthly_payment 함수 유지) ...
    if annual_rate <= 0 or years <= 0:
        return principal / max(years * 12, 1)
    r = annual_rate / 12
    n = years * 12
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def compute_dsr_eligibility(
        annual_income: float,
        desired_loan: float,
        rate: float,
        term_years: int,
        dsr_limit: float = 0.4,  # 은행권 DSR 40% 기준
        ) -> Tuple[float, float]:

    # ... (기존 compute_dsr_eligibility 함수 유지) ...
    if annual_income <= 0 or term_years <= 0 or rate <= 0:
        return 0.0, 0.0
    monthly_payment = calc_monthly_payment(desired_loan, rate, term_years)
    annual_debt_service = monthly_payment * 12
    dsr = annual_debt_service / annual_income
    max_annual_debt_service = annual_income * dsr_limit
    r = rate / 12
    n = term_years * 12
    monthly_installment_factor = (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    if monthly_installment_factor == 0:
        max_loan_dsr = 0.0
    else:
        max_loan_dsr = max_annual_debt_service / (monthly_installment_factor * 12)
    return dsr, max_loan_dsr


##############################################################################################################
# --- 3. SQLite 데이터 추출 및 가공 ---
##############################################################################################################

# --- 정규표현식: 요청된 6가지 특정 금리 추출용 ---
# 1. locked_rate_scheme 전체 범위 추출 (고정금리)
RE_LOCKED_RANGE = re.compile(
    r'locked_rate_scheme\s*:\s*(?:floor_rate_value\s*(?P<min>\d{1,2}(?:\.\d+)?)\s*%)'
    r'[^~]*~[^~]*'
    r'(?:ceiling_rate_value\s*(?P<max>\d{1,2}(?:\.\d+)?)\s*%)', re.I | re.S)

# 2. adaptive_rate_scheme 전체 범위 추출 (변동금리)
RE_ADAPTIVE_RANGE = re.compile(
    r'adaptive_rate_scheme\s*:\s*(?:floor_rate_value\s*(?P<min>\d{1,2}(?:\.\d+)?)\s*%)'
    r'[^~]*~[^~]*'
    r'(?:ceiling_rate_value\s*(?P<max>\d{1,2}(?:\.\d+)?)\s*%)', re.I | re.S)

# LTV 정규식 (유지)
RE_LTV = re.compile(r'(LTV|담보인정비율|최대\s*적용\s*가능\s*LTV)[^0-9]{0,15}(?P<ltv>\d{1,3})\s*%(\s*(?!p))?', re.I)

###########################
# --- 제출 서류 추출 정규 표현식 및 함수 ---
# 필요한 서류 키워드: 필요서류, 제출서류, 관련서류, 대출준비서류, 선택서류
# \s*[:\s]*: 키워드 뒤의 구분자 (공백, 콜론 등)
# (.*?): 내용 (비탐욕적)
# (?:\n\n|\n\s*(?:필요서류|제출서류|관련서류|대출준비서류|선택서류)|$): 다음 빈 줄, 다음 키워드, 또는 텍스트 끝이 나올 때까지 캡처
RE_DOCS = re.compile(
    r'(필요서류|제출서류|관련서류|대출준비서류|선택서류)\s*[:\s]*(.*?)(?:\n\n|\n\s*(?:필요서류|제출서류|관련서류|대출준비서류|선택서류)|$)',
    re.I | re.S
)

##############


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """필수 컬럼 확인 및 타입 조정"""
    RATE_COLS = ["locked_min", "locked_max", "adaptive_min", "adaptive_max", "max_ltv"]
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = df.get(col, "" if col not in RATE_COLS else float("nan"))
    for col in RATE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _clean_rate(v: Optional[float]) -> Optional[float]:
    """금리 유효성 검사 및 정제."""
    if v is None: return None
    if v < 0.5 or v > 30: return None
    return v


def extract_fixed_variable_rates_and_ltv(text: str) -> Tuple[
    Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """텍스트에서 고정금리 범위, 변동금리 범위, LTV 값을 추출합니다.
    (locked_min, locked_max, adaptive_min, adaptive_max, ltv)"""

    locked_min = locked_max = adaptive_min = adaptive_max = ltv = None
    if not text:
        return locked_min, locked_max, adaptive_min, adaptive_max, ltv

    t = re.sub(r'\s+', ' ', text)

    # 1. locked_rate_scheme 범위 추출 (고정금리)
    m = RE_LOCKED_RANGE.search(t)
    if m:
        locked_min = _clean_rate(float(m.group('min')))
        locked_max = _clean_rate(float(m.group('max')))
        if locked_min is not None and locked_max is not None and locked_min > locked_max:
            locked_min, locked_max = locked_max, locked_min

    # 2. adaptive_rate_scheme 범위 추출 (변동금리)
    m = RE_ADAPTIVE_RANGE.search(t)
    if m:
        adaptive_min = _clean_rate(float(m.group('min')))
        adaptive_max = _clean_rate(float(m.group('max')))
        if adaptive_min is not None and adaptive_max is not None and adaptive_min > adaptive_max:
            adaptive_min, adaptive_max = adaptive_max, adaptive_min

    # 3. LTV 추출 (기존 로직 유지)
    ml = RE_LTV.search(t)
    if ml:
        try:
            ltv = float(ml.group('ltv'))
            if ltv is not None and (ltv < 10 or ltv > 100): ltv = None
        except Exception:
            ltv = None

    return locked_min, locked_max, adaptive_min, adaptive_max, ltv


def fetch_rows(conn: sqlite3.Connection, table: str) -> Dict[str, Any]:
    """DB 테이블에서 모든 행을 가져와 raw_text를 만듭니다."""
    cur = conn.cursor()
    cur.execute(f"SELECT rowid, data_value FROM \"{table}\"")
    full_text = [str(row[1]) for row in cur.fetchall()]
    # table_name = table.replace('_data', '').strip()
    table_name = table.replace('_data', '').replace('_terms', '').strip()
    return {
        "bank_name": None,
        "source_table": table_name,
        "raw_text": "\n".join(full_text),
        "term_years": None,
    }


def fetch_all_products(conn: sqlite3.Connection):
    """DB의 모든 테이블에서 데이터를 추출하여 상품 리스트를 만듭니다."""
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [tbl[0] for tbl in cur.fetchall()]

    all_products = []
    for table_name in tables:
        if table_name.lower() == 'sqlite_sequence': continue
        try:
            # fetch_rows가 raw_text를 반환하도록 수정됨
            product_data = fetch_rows(conn, table_name)
            all_products.append(product_data)
        except Exception:
            continue
    return all_products


def guess_bank_and_product(bank_name: str, source_table: str):
    """테이블 이름에서 은행명과 상품명을 유추합니다."""
    src = (source_table or "").strip()
    parts = src.split('_', 1)
    bank, product = (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (src.strip(), "약관")
    bank = re.sub(r'(은행|저축은행|주식회사)$', '', bank).strip()
    product = product.replace("상품설명서", "").replace("상품약관", "").strip()
    return bank or "미상", product or "약관"


def detect_loan_type(text: str, product: str) -> str:
    """텍스트에서 대출 유형(전세, 정책형, 주담대)을 유추합니다."""
    t = f"{text or ''} {product or ''}".lower()
    if any(k in t for k in ["전세", "jeonse", "lease"]): return "전세"
    if any(k in t for k in ["디딤돌", "버팀목", "생애최초", "보금자리", "특례보금자리"]): return "정책형"
    if any(k in t for k in ["주담대", "주택담보", "모기지", "담보대출", "주택구입", "아파트", "오피스텔"]): return "주담대"
    return "주담대"


def build_df_from_sqlite(db_path: str) -> pd.DataFrame:
    """SQLite DB에서 데이터를 추출, 가공하여 DataFrame을 직접 반환합니다."""
    if not os.path.exists(db_path):
        # NOTE: 원본 코드의 파일 오류 예외 처리를 따름
        # DB가 없는 경우 추천 시스템에서 더미 데이터로 대체할 수 있도록 에러 발생 대신 빈 DataFrame 반환
        return pd.DataFrame(columns=REQUIRED_COLS)

    conn = sqlite3.connect(db_path)
    try:
        products_data = fetch_all_products(conn)
    finally:
        conn.close()

    if not products_data:
        return pd.DataFrame(columns=REQUIRED_COLS)

    out = []
    for r in products_data:
        bank, product = guess_bank_and_product(r.get("bank_name"), r.get("source_table"))
        raw_text = r.get("raw_text") or ""

        # 새로운 추출 함수 사용 및 컬럼 매핑
        locked_min, locked_max, adaptive_min, adaptive_max, ltv = extract_fixed_variable_rates_and_ltv(raw_text)

        loan_type = detect_loan_type(raw_text, product)
        doc_text = raw_text.strip()

        documents_required = extract_document_info(doc_text)  ###################-----------------------

        out.append({
            "bank": bank,
            "product": product,
            "loan_type": loan_type,
            # 새로운 금리 컬럼 사용
            "locked_min": locked_min,
            "locked_max": locked_max,
            "adaptive_min": adaptive_min,
            "adaptive_max": adaptive_max,
            # --- [수정 끝] ---
            "max_ltv": ltv,
            "eligible_rule": "",
            "advantage_tips": "",
            "doc_path": f"db://{r.get('source_table')}/{bank}_{product}",
            "doc_text": doc_text[:8000],
            "documents_required": documents_required  ######################--------------
        })

    df = pd.DataFrame(out)
    df = _ensure_columns(df)
    return df


# --- 사용자 선호 금리 유형에 따른 금리 범위 포맷팅 함수 ---
def _fmt_rate(r: pd.Series, user_pref: Literal["고정", "변동"]) -> str:
    """사용자 선호에 따라 고정/변동 금리 정보를 보기 좋은 문자열로 변환합니다."""
    prefix = "locked" if user_pref == "고정" else "adaptive"
    rmin = r.get(f"final_rate_min") if r.get(f"final_rate_min") is not None else r.get(f"{prefix}_min")
    rmax = r.get(f"final_rate_max") if r.get(f"final_rate_max") is not None else r.get(f"{prefix}_max")

    if pd.notna(rmin) and pd.notna(rmax): return f"{rmin:.2f}%~{rmax:.2f}%"
    if pd.notna(rmin): return f"{rmin:.2f}%"
    if pd.notna(rmax): return f"{rmax:.2f}%"
    return f"자료상 {user_pref}금리 미기재"


def _safe_parse_json(s: str) -> dict:
    """JSON 문자열을 안전하게 파싱합니다."""
    s = s.strip()
    s = re.sub(r'^```(?:json)?', '', s, flags=re.I).strip()
    s = re.sub(r'```$', '', s, flags=re.I).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r'\{.*\}', s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


##############################################################################################################
# --- 4. 추천 시스템 메인 로직 ---
##############################################################################################################
# 너는 한국의 은행 주택담보대출 전문 상담 직원이며, 대출 상품을 과도하게 권유하지 않고 사용자의 입력 정보와 수치 데이터를 기반으로 차분하고 신뢰감 있는 설명을 제공해야 한다.
# 사용자의 상황에 맞춰 해당 상품을 추천하는 이유를 한국어로 정확히 3문장으로 작성하되, 각 문장은 100자 이내, 전체는 300자 이내로 제한한다.
# 1문장에서는 상품이 사용자의 조건과 목적에 어떻게 부합하는지 핵심을 요약하고,
# 2문장에서는 금리·최대 대출 가능액·월 상환액·DSR·LTV·규제지역 여부 등 제공된 수치와 정책 기준을 근거로 장점을 설명하며
# 가능하면 “이 상품은 규제지역 기준 LTV 40%가 적용되어 최대 ○○억까지 가능하여…”처럼 자연스럽게 포함한다.
# 3문장에서는 상환 부담 변화나 금리 변동 가능성 등 사용자가 알아야 할 주의사항을 간단히 언급하고,
# 3문장을 말한 뒤에는 절대 추가 설명을 하지 않으며 마지막 문장은 반드시 마침표로 끝낸다.
# 모든 설명은 제공된 수치와 약관 정보만을 근거로 하며, 추측·과장·정책 확정 표현·과도한 영업 멘트는 절대 포함하지 않는다.
# 반드시 유효한 JSON 형식으로만 응답하세요.
#
# [사용자]{user_json}
# [후보]
# {top_rows}
#
# 출력 형식:
# {{"recommendations":[
#     {{"bank":"OO","product":"...", "rate_range":"3.5%~4.8%","why":"..."}},
#     ...
# ]}}

# 사용자 정보와 상위 후보 상품 정보를 바탕으로 최적의 주택담보대출 3개를 추천합니다.
# 금리 경쟁력, 예상 DSR 충족 여부, 최대 대출 가능 금액을 고려하여 추천 이유(why)를 2~3문장으로 작성하고,


# 주택담보대출 상품을 과도하게 권유하지 않고 사용자의 입력 정보와 수치 데이터를 기반으로 차분하고 신뢰감 있는 설명을 제공해야 합니다.
# 사용자의 상황에 맞춰 해당 상품을 추천하는 이유를 한국어로 정확히 3문장으로 작성하고, 총 글자수는 300자 이내로 제한합니다.
#
# 첫번째 문장에서는 상품이 사용자의 조건과 목적에 어떻게 부합하는지 핵심을 요약해야 합니다.
#
# 두번째 문장에서는 금리·최대 대출 가능액·월 상환액·DSR·LTV·규제지역 여부 등 제공된 수치와 정책 기준을 근거로 장점을 설명하며
# 가능하면 “이 상품은 규제지역 기준 LTV 40%가 적용되어 최대 ○○억까지 가능하여…”처럼 자연스럽게 포함해야 합니다.
#
# 세번째 문장에서는 상환 부담 변화나 금리 변동 가능성 등 사용자가 알아야 할 주의사항을 간단히 언급하고,
# 세번째 문장을 말한 뒤에는 절대 추가 설명을 하지 않아야 합니다.
#
# 모든 설명은 제공된 수치와 약관 정보만을 근거로 하며, 추측·과장·정책 확정 표현·과도한 영업 멘트는 절대 포함하지 않아야 합니다.
# 반드시 유효한 JSON 형식으로만 응답해야 합니다.


# 당신은 금융 전문가입니다.
#
# [요구사항]
# 1.  반드시 한국어 3문장으로 작성해야 합니다. 각 문장은 100자 이내로 구성해야 합니다.
# 2.  추측, 과장, 과도한 영업 멘트는 절대 금지하며, 제공된 수치와 정책 기반으로만 설명합니다.
#
# [문장별 지침]
# * **첫 번째 문장:** 상품이 사용자의 주택구입 목적, 선호 기간(term_years) 등 조건에 부합하는 이유를 핵심적으로 요약합니다.
# * **두 번째 문장:** 금리(`rate_range`), 최대 대출 가능액(`max_possible_loan`), DSR(`estimated_dsr`) 등 수치를 인용하여
#                    상품의 장점을 구체적으로 설명합니다. 특히 규제지역 여부와 LTV 정책을 언급해야 합니다.
# * **세 번째 문장:** 상환 부담 변화나 금리 변동 가능성 등 **사용자가 알아야 할 주의사항**을 간결하게 언급하고 설명을 종료합니다.

######################

def extract_document_info(text: str) -> str:
    """상품 텍스트에서 제출 서류 관련 정보를 추출합니다."""
    if not text:
        return "제출 서류 정보 없음."

    # 줄 바꿈 통일
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    extracted_info = []

    # 모든 일치 항목을 찾습니다.
    matches = RE_DOCS.findall(text)

    for keyword, content in matches:
        content_clean = content.strip()

        # 내용이 너무 길면 잘라줍니다. (최대 500자)
        if len(content_clean) > 500:
            content_clean = content_clean[:500] + "..."#############################-----------------

        if content_clean:
            # 추출된 정보를 '키워드: 내용' 형식으로 저장
            extracted_info.append(f"**{keyword}**: {content_clean}")

    if extracted_info:
        return "\n".join(extracted_info)
    else:
        return "제출 서류 정보 없음."

###############################




EXPLAIN_PROMPT = """

너는 한국의 은행 주택담보대출 전문 상담 직원이며, 대출 상품을 과도하게 권유하지 않고 사용자의 입력 정보와 수치 데이터를 기반으로 차분하고 신뢰감 있는 설명을 제공해야 한다.
사용자의 상황에 맞춰 해당 상품을 추천하는 이유를 한국어로 정확히 3문장으로 작성하되, 각 문장은 100자 이내, 전체는 300자 이내로 제한한다.
1문장에서는 상품이 사용자의 조건과 목적에 어떻게 부합하는지 핵심을 요약하고,
2문장에서는 금리·최대 대출 가능액·월 상환액·DSR·LTV·규제지역 여부 등 제공된 수치와 정책 기준을 근거로 장점을 설명하며
가능하면 “이 상품은 규제지역 기준 LTV 40%가 적용되어 최대 ○○억까지 가능하여…”처럼 자연스럽게 포함한다.
3문장에서는 상환 부담 변화나 금리 변동 가능성 등 사용자가 알아야 할 주의사항을 간단히 언급하고,
3문장을 말한 뒤에는 절대 추가 설명을 하지 않으며 마지막 문장은 반드시 마침표로 끝낸다.
모든 설명은 제공된 수치와 약관 정보만을 근거로 하며, 추측·과장·정책 확정 표현·과도한 영업 멘트는 절대 포함하지 않는다.
반드시 유효한 JSON 형식으로만 응답하세요.

[사용자]{user_json}
[후보]
{top_rows}

출력 형식:
{{"recommendations":[
    {{"bank":"OO","product":"...", "rate_range":"3.5%~4.8%","why":"..."}},
    ...
]}}

"""

# def rule_filter(df: pd.DataFrame, u: UserInput, rc: List[ReasoningStep]) -> pd.DataFrame:
#     """
#     요청된 LTV, DSR, 상품별 조건(나이/고용/담보/생애최초)을 적용하여 하드 필터링합니다.
#     UserInput -> "annual_income, desired_loan_amount, loan_term_years, is_first_homebuyer"
#     """
#
#     if df.empty:
#         rc.append(ReasoningStep(step="상품 데이터 확인 필터",
#                                 detail="데이터프레임이 비어 있어 필터를 통과합니다."))
#         return df.copy()
#
#     ###
#     # LTV 정책 한도 (LTV 한도와 법정 최대한도 중 작은 값)
#     max_loan_ltv_cap, ltv_reason, policy_ltv_rate = compute_policy_ltv_and_reason(u)
#     rc.append(ReasoningStep(step=" LTV 정책 한도", detail=ltv_reason))
#     ###
#     out = df.copy()
#
#     mask_type = out["loan_type"].astype(str).str.contains("주담대|정책형", na=False)
#     # rc.append(ReasoningStep(step=" 상품 유형 필터",
#     #                         detail=f"주택 담보 대출 및 정책형 상품만 필터링 → {len(out)}건"))
#     out = out[mask_type]
#
#     if out.empty:
#         # rc.append(ReasoningStep(step="상품 유형 필터",
#         #                         detail=ltv_reason + " 주담대/정책형 상품이 없어 필터링을 통과합니다."))
#         return out
#
#     # 1-1. 정책 LTV 필터
#     if u.desired_loan_amount > max_loan_ltv_cap:
#         reason_policy_ltv = (
#             f"원하는 대출금액({u.desired_loan_amount:,.0f}원)이 정책 LTV/CAP 한도 "
#             f"({max_loan_ltv_cap:,.0f}원)를 초과하여 모든 상품 후보에서 제외됩니다."
#         )
#         out = out.iloc[0:0]
#     else:
#         reason_policy_ltv = f"원하는 대출금액이 정책 LTV/CAP 한도({max_loan_ltv_cap:,.0f}원)를 충족하여 상품 유지."
#
#     # 1-2. 상품 LTV 필터 (정책 LTV 통과한 경우에만)
#     if not out.empty:
#         ltv_max_prod = pd.to_numeric(out["max_ltv"], errors='coerce').fillna(100)
#         product_max_loan_by_ltv = ltv_max_prod * u.house_price / 100.0
#         mask_prod_ltv = (u.desired_loan_amount <= product_max_loan_by_ltv)
#         products_before = len(out)
#         out = out[mask_prod_ltv]
#         reason_prod_ltv = f"상품별 LTV 기준 한도 초과 상품 제외 → {len(out)}건 (총 {products_before - len(out)}건 제외)."
#     else:
#         reason_prod_ltv = "정책 LTV 필터에서 모두 제외되어 상품 LTV 필터는 실행되지 않았습니다."
#
#     # LTV 관련 모든 단계를 하나로 결합
#     combined_ltv_detail = f"**[정책 LTV 한도]**\n{ltv_reason}\n" \
#                           f"\n**[정책 LTV 필터 결과]**\n{reason_policy_ltv}\n" \
#                           f"\n**[상품 LTV 필터 결과]**\n{reason_prod_ltv}"
#
#     if out.empty: return out
#
#     # DSR 한도 계산 (평균 5% 금리, 40% DSR 기준 가정)
#     # DSR 필터링을 위한 최대 가능 대출 금액 (Desired Loan Amount 기준)
#     rate_for_dsr_check = 0.05
#     dsr_current, max_loan_dsr = compute_dsr_eligibility(
#         annual_income=u.annual_income,
#         desired_loan=u.desired_loan_amount,
#         rate=rate_for_dsr_check,
#         term_years=u.loan_term_years,
#         dsr_limit=0.4
#     )
#     rc.append(ReasoningStep(step="0. DSR 사전 계산",
#                             detail=f"DSR({dsr_current:.1%}) 기준 최대 가능액: {max_loan_dsr:,.0f}원"))
#     out = df.copy()
#
#
#     # 2-1. DSR 필터
#     if u.desired_loan_amount > max_loan_dsr:
#         reason_dsr_filter = (
#             f"원하는 대출금액({u.desired_loan_amount:,.0f}원)이 DSR 한도 기준 최대 가능액 "
#             f"({max_loan_dsr:,.0f}원)을 초과하여 모든 상품에서 제외됩니다."
#         )
#         out = out.iloc[0:0]
#     else:
#         reason_dsr_filter = f"원하는 대출금액이 DSR 한도를 충족하여 상품 유지."
#
#     # DSR 관련 모든 단계를 하나로 결합
#     combined_dsr_detail = (
#         f"**[DSR 사전 계산]**\n"
#         f"DSR({dsr_current:.1%}) 기준 최대 가능액: {max_loan_dsr:,.0f}원\n\n"
#         f"**[DSR 필터 결과]**\n"
#         f"{reason_dsr_filter}"
#     )
#     rc.append(ReasoningStep(step="DSR 종합 필터", detail=combined_dsr_detail))
#
#     if out.empty: return out
#     # --- 2. DSR (사전 계산 + 필터) 결합 로직 끝 ---
#
#     if "is_first_time_required" in out.columns:
#         mask_first = ~((out["is_first_time_required"] == True) & (u.is_first_homebuyer == False))
#         out = out[mask_first]
#         # rc.append(ReasoningStep(step="5. 생애최초 필터", detail=f"생애최초 요건 상품 필터링 후 {len(out)}건"))
#
#     return out.copy()

# def rule_filter(df: pd.DataFrame, u: UserInput, rc: List[ReasoningStep]) -> pd.DataFrame:
#     """
#     요청된 LTV, DSR, 상품별 조건(나이/고용/담보/생애최초)을 적용하여 하드 필터링합니다.
#     UserInput -> "annual_income, desired_loan_amount, loan_term_years, is_first_homebuyer"
#     """
#     if df.empty:
#         rc.append(ReasoningStep(step="1차 필터", detail="데이터프레임이 비어 있어 필터링을 건너뜜니다."))
#         return df.copy()
#
#     # --- 0. 정책 한도 사전 계산 ---
#     # LTV 정책 한도 (LTV 한도와 법정 최대한도 중 작은 값)
#     max_loan_ltv_cap, ltv_reason, policy_ltv_rate = compute_policy_ltv_and_reason(u)
#     rc.append(ReasoningStep(step="0. 정책 LTV 한도", detail=ltv_reason))
#
#     # DSR 한도 계산 (평균 5% 금리, 40% DSR 기준 가정)
#     # DSR 필터링을 위한 최대 가능 대출 금액 (Desired Loan Amount 기준)
#     rate_for_dsr_check = 0.05
#     dsr_current, max_loan_dsr = compute_dsr_eligibility(
#         annual_income=u.annual_income,
#         desired_loan=u.desired_loan_amount,
#         rate=rate_for_dsr_check,
#         term_years=u.loan_term_years,
#         dsr_limit=0.4
#     )
#     rc.append(ReasoningStep(step="0. DSR 사전 계산",
#                             detail=f"DSR({dsr_current:.1%}) 기준 최대 가능액: {max_loan_dsr:,.0f}원"))
#
#     out = df.copy()
#
#     # --- 1. 대출 유형 필터 (주담대/정책형만) ---
#     # ⚠️ u.purpose 사용을 완전히 제거하고 주택 담보 대출 목적 상품만 필터링합니다.
#     mask_type = out["loan_type"].astype(str).str.contains("주담대|정책형", na=False)
#     out = out[mask_type]
#     # rc.append(ReasoningStep(step="1. 유형 필터", detail=f"주택 담보 대출 및 정책형 상품만 필터링 → {len(out)}건"))
#     if out.empty: return out
#
#     # --- 2. 정책 LTV 한도 필터 (대출 금액 기준) ---
#     # 원하는 대출 금액이 정책 LTV/CAP 한도를 초과하는 경우, 모든 상품에서 드롭 (대출 불가)
#     if u.desired_loan_amount > max_loan_ltv_cap:
#         reason = (
#             f"원하는 대출금액({u.desired_loan_amount:,.0f}원)이 정책 LTV/CAP 한도 "
#             f"({max_loan_ltv_cap:,.0f}원)를 초과하여 모든 상품 후보에서 제외됩니다."
#         )
#         out = out.iloc[0:0]
#     else:
#         reason = f"원하는 대출금액이 정책 LTV/CAP 한도를 충족하여 상품 유지."
#
#     # rc.append(ReasoningStep(step="2. 정책 LTV 필터", detail=reason))
#     if out.empty:
#         return out
#
#     # --- DSR 필터 로직 및 추론 단계 기록 ---
#     # DSR 한도 계산 결과 (max_loan_dsr)는 '0. DSR 사전 계산' 단계에서 이미 계산되어 있어야 합니다.
#
#     if u.desired_loan_amount > max_loan_dsr:
#         reason_dsr = (
#             f"원하는 대출금액({u.desired_loan_amount:,.0f}원)이 DSR 한도 기준 최대 가능액 "
#             f"({max_loan_dsr:,.0f}원)을 초과하여 모든 상품에서 제외됩니다."
#         )
#         out = out.iloc[0:0]
#     else:
#         # out은 LTV 필터를 통과한 상품 리스트입니다.
#         reason_dsr = f"원하는 대출금액이 DSR 한도를 충족하여 상품 유지. ({len(out)}건)"
#
#     # 추론 단계에 DSR 필터 결과 기록
#     rc.append(ReasoningStep(step="DSR 필터", detail=reason_dsr))
#
#     if out.empty:
#         return out
#
#     # --- 3. 상품별 최대 LTV 한도 필터 ---
#     # 상품 DB에 max_ltv(%) 컬럼이 있어야 함
#     # 이 상품의 LTV 기준 최대 대출 가능 금액
#     # pd.to_numeric()을 사용하여 max_ltv가 없는(NaN) 상품도 통과시키도록 처리
#     ltv_max_prod = pd.to_numeric(out["max_ltv"], errors='coerce').fillna(100)  # LTV 정보 없으면 100%로 간주 (통과)
#
#     product_max_loan_by_ltv = ltv_max_prod * u.house_price / 100.0
#
#     # 원하는 대출 금액이 상품의 LTV 기준 최대 대출 금액을 초과하는 경우 제외
#     mask_prod_ltv = (u.desired_loan_amount <= product_max_loan_by_ltv)
#     out = out[mask_prod_ltv]
#     rc.append(ReasoningStep(step="3. 상품 LTV 필터", detail=f"상품 LTV 기준 한도 초과 상품 제외 → {len(out)}건"))
#     if out.empty:
#         return out
#
#     # --- 4. DSR 한도 필터 ---
#     # 원하는 대출 금액 기준으로 계산된 DSR이 정책 한도(0.4)를 초과하는 경우 제외
#     # ⚠️ DTI 필터는 DSR에 포함되거나 별도 부채 정보가 필요하므로 DSR 필터만 적용합니다.
#     if u.desired_loan_amount > max_loan_dsr:
#         reason_dsr = (
#             f"원하는 대출금액({u.desired_loan_amount:,.0f}원)이 DSR 한도 기준 최대 가능액 "
#             f"({max_loan_dsr:,.0f}원)을 초과하여 모든 상품에서 제외됩니다."
#         )
#         out = out.iloc[0:0]
#     else:
#         reason_dsr = f"원하는 대출금액이 DSR 한도를 충족하여 상품 유지."
#
#     # rc.append(ReasoningStep(step="4. DSR 필터", detail=reason_dsr))
#     if out.empty: return out
#
#     # --- 5. 상품별 세부 조건 필터 ---
#
#     # 5-1. 생애 최초 요구 상품 필터
#     # DB에 'is_first_time_required' 컬럼이 있다고 가정
#     if "is_first_time_required" in out.columns:
#         mask_first = ~(
#                 (out["is_first_time_required"] == True) &
#                 (u.is_first_homebuyer == False)
#         )
#         out = out[mask_first]
#         rc.append(ReasoningStep(step="5-1. 생애최초 필터",
#                                 detail=f"생애최초 요구 조건 불충족 상품 제외 → {len(out)}건"))
#
#     return out.copy()
#####################################################################
#####################################################################

def rule_filter(df: pd.DataFrame, u: UserInput, rc: List[ReasoningStep]) -> pd.DataFrame:
    """
    요청된 LTV, DSR, 상품별 조건(나이/고용/담보/생애최초)을 적용하여 하드 필터링합니다.
    UserInput -> "annual_income, desired_loan_amount, loan_term_years, is_first_homebuyer"
    """
    if df.empty:
        rc.append(ReasoningStep(step="1차 필터", detail="데이터프레임이 비어 있어 필터링을 건너뜜니다."))
        return df.copy()

    # DSR 한도 계산 (평균 5% 금리, 40% DSR 기준 가정)
    # DSR 필터링을 위한 최대 가능 대출 금액 (Desired Loan Amount 기준)
    rate_for_dsr_check = 0.05
    dsr_current, max_loan_dsr = compute_dsr_eligibility(
        annual_income=u.annual_income,
        desired_loan=u.desired_loan_amount,
        rate=rate_for_dsr_check,
        term_years=u.loan_term_years,
        dsr_limit=0.4
    )
    # rc.append(ReasoningStep(step="정책 DSR 필터",
    #                         detail=f"DSR({dsr_current:.1%}) 기준 최대 가능액: {max_loan_dsr:,.0f}원"))

    out = df.copy()

    # --- 0. 정책 한도 사전 계산 ---
    # LTV 정책 한도 (LTV 한도와 법정 최대한도 중 작은 값)
    max_loan_ltv_cap, ltv_reason, policy_ltv_rate = compute_policy_ltv_and_reason(u)
    # rc.append(ReasoningStep(step="0. 정책 LTV 한도", detail=ltv_reason))

    # --- 1. 대출 유형 필터 (주담대/정책형만) ---
    # ⚠️ u.purpose 사용을 완전히 제거하고 주택 담보 대출 목적 상품만 필터링합니다.
    # mask_type = out["loan_type"].astype(str).str.contains("주담대|정책형", na=False)
    # out = out[mask_type]
    # if out.empty: return out

    # --- 2. 정책 LTV 한도 필터 (대출 금액 기준) ---
    # 원하는 대출 금액이 정책 LTV/CAP 한도를 초과하는 경우, 모든 상품에서 드롭 (대출 불가)
    if u.desired_loan_amount > max_loan_ltv_cap:
        reason = (
            f"원하는 대출금액({u.desired_loan_amount:,.0f}원)이 정책 LTV/CAP 한도 "
            f"({max_loan_ltv_cap:,.0f}원)를 초과하여 모든 상품 후보에서 제외됩니다."
        )
        out = out.iloc[0:0]
    else:
        reason = f"원하는 대출금액이 정책 LTV/CAP 한도를 충족하여 상품 유지."

    rc.append(ReasoningStep(step="정책 LTV 필터", detail=reason))
    if out.empty:
        return out
    #####################################################################

    # --- 3. 상품별 최대 LTV 한도 필터 ---
    # 상품 DB에 max_ltv(%) 컬럼이 있어야 함
    # 이 상품의 LTV 기준 최대 대출 가능 금액
    # pd.to_numeric()을 사용하여 max_ltv가 없는(NaN) 상품도 통과시키도록 처리
    ltv_max_prod = pd.to_numeric(out["max_ltv"], errors='coerce').fillna(100)  # LTV 정보 없으면 100%로 간주 (통과)

    product_max_loan_by_ltv = ltv_max_prod * u.house_price / 100.0

    # 원하는 대출 금액이 상품의 LTV 기준 최대 대출 금액을 초과하는 경우 제외
    mask_prod_ltv = (u.desired_loan_amount <= product_max_loan_by_ltv)
    out = out[mask_prod_ltv]
    rc.append(ReasoningStep(step="상품 LTV 필터", detail=f"상품 LTV 기준 한도 초과 상품 제외, {len(out)}건 유지"))
    if out.empty: return out
    #####################################################################

    # --- 4. DSR 한도 필터 ---
    # 원하는 대출 금액 기준으로 계산된 DSR이 정책 한도(0.4)를 초과하는 경우 제외
    # ⚠️ DTI 필터는 DSR에 포함되거나 별도 부채 정보가 필요하므로 DSR 필터만 적용합니다.
    if u.desired_loan_amount > max_loan_dsr:
        reason_dsr = (
            f"원하는 대출금액({u.desired_loan_amount:,.0f}원)이 DSR 한도 기준 최대 가능액 "
            f"({max_loan_dsr:,.0f}원)을 초과하여 모든 상품에서 제외됩니다."
        )
        out = out.iloc[0:0]
    else:
        reason_dsr = f"원하는 대출금액이 DSR 한도를 충족하여 상품 유지."

    # rc.append(ReasoningStep(step="상품 DSR 필터", detail=reason_dsr))
    rc.append(ReasoningStep(step="사용자 데이터 맞춤 DSR 필터", detail=reason_dsr))
    if out.empty:
        return out

    #####################################################################




    return out.copy()
#####################################################################
#####################################################################

def score_rule(product: pd.Series, user: UserInput, loan_amount: float) -> float:
    """
    요청된 4가지 기준으로 상품에 점수를 부여합니다.
    """
    score = 0.0

    # 1) 금리 점수 (낮을수록 점수↑, 50% 가중치)
    user_pref = user.fixed_or_variable_pref
    prefix = "locked" if user_pref == "고정" else "adaptive"

    # 선호 금리 사용 시도
    rmin = product.get(f"{prefix}_min")
    rmax = product.get(f"{prefix}_max")

    # 선호 금리가 없으면, 다른 쪽 금리를 대체로 사용
    if pd.isna(rmin) and pd.isna(rmax):
        other_prefix = "adaptive" if user_pref == "고정" else "locked"
        rmin = product.get(f"{other_prefix}_min")
        rmax = product.get(f"{other_prefix}_max")

    rmin = rmin if pd.notna(rmin) else 5.0
    rmax = rmax if pd.notna(rmax) else 5.0

    avg_rate = (rmin + rmax) / 2
    rate_score = max(0, 100 - avg_rate * 10)
    score += rate_score * 0.5  # 50% 가중치

    # 2) 대출 한도 점수 (원하는 금액 대비 여유가 있는지, 20% 가중치)
    ltv_max_prod = product.get("max_ltv") if pd.notna(product.get("max_ltv")) and product.get("max_ltv") > 0 else 100
    max_loan_by_ltv = ltv_max_prod * user.house_price / 100.0

    limit_ratio = max_loan_by_ltv / loan_amount
    if limit_ratio < 1:
        limit_score = 0
    else:
        limit_score = min(100, (limit_ratio - 1) * 200)
    score += limit_score * 0.2  # 20% 가중치

    # 3) 선호 금리 유형 (고정/변동) 적합도 (15% 가중치)
    # 선호하는 금리 유형의 데이터가 존재하는지 확인
    user_pref_prefix = "locked" if user.fixed_or_variable_pref == "고정" else "adaptive"

    if pd.notna(product.get(f"{user_pref_prefix}_min")) or pd.notna(product.get(f"{user_pref_prefix}_max")):
        pref_score = 100  # 선호하는 금리 데이터가 있으면 100점
    else:
        # 선호하지 않는 금리 유형의 데이터라도 있다면 50점 (차선책 존재)
        other_pref_prefix = "adaptive" if user_pref_prefix == "locked" else "locked"
        if pd.notna(product.get(f"{other_pref_prefix}_min")) or pd.notna(product.get(f"{other_pref_prefix}_max")):
            pref_score = 50
        else:
            pref_score = 0
    score += pref_score * 0.15  # 15% 가중치

    # 4) 상환기간 적합도 (15% 가중치)
    min_p = product.get("min_period") if pd.notna(product.get("min_period")) else 10
    max_p = product.get("max_period") if pd.notna(product.get("max_period")) else 40
    user_period = user.loan_term_years

    if min_p <= user_period <= max_p:
        period_score = 100
    else:
        period_score = 0
    score += period_score * 0.15  # 15% 가중치

    return score


def score_products(df: pd.DataFrame, u: UserInput, rc: List[ReasoningStep]) -> pd.DataFrame:
    """필터링된 상품에 스코어를 매기고, 예상 지표를 추가합니다."""
    if df.empty: return df

    max_loan_ltv_cap, ltv_reason, policy_ltv = compute_policy_ltv_and_reason(u)
    rc.append(ReasoningStep(step="정책 LTV 한도", detail=ltv_reason))

    rate_for_strict_dsr_check = 0.05
    _, max_loan_dsr_strict = compute_dsr_eligibility(
        annual_income=u.annual_income,
        desired_loan=1.0,
        rate=rate_for_strict_dsr_check,
        term_years=u.loan_term_years,
        dsr_limit=0.4
    )
    rc.append(ReasoningStep(step="정책 DSR 한도",
                            detail=f"스코어링을 위한 DSR 40% 기준 최대 가능액: {max_loan_dsr_strict:,.0f}원"))

    out = df.copy()

    def apply_metrics_and_score(r):
        # 금리 값 결정 로직: 사용자 선호에 따라 금리 추출
        user_pref = u.fixed_or_variable_pref
        prefix = "locked" if user_pref == "고정" else "adaptive"

        # 선호 금리 사용 시도
        rmin = r.get(f"{prefix}_min")
        rmax = r.get(f"{prefix}_max")

        # 선호 금리가 없으면, 다른 금리 사용 (계산은 해야 하므로)
        if pd.isna(rmin) and pd.isna(rmax):
            other_prefix = "adaptive" if user_pref == "고정" else "locked"
            rmin = r.get(f"{other_prefix}_min")
            rmax = r.get(f"{other_prefix}_max")

        # 둘 다 없으면 기본값 5.0%
        rmin = rmin if pd.notna(rmin) else 5.0
        rmax = rmax if pd.notna(rmax) else 5.0
        rate_for_calc = (rmin + rmax) / 2.0
        # --- [수정 끝] ---

        ltv_max_prod = r.get("max_ltv")

        # DSR 계산 (은행권 DSR 40% 적용)
        dsr_limit = 0.4
        dsr, _ = compute_dsr_eligibility(
            annual_income=u.annual_income,
            desired_loan=u.desired_loan_amount,
            rate=rate_for_calc / 100.0,
            term_years=u.loan_term_years,
            dsr_limit=dsr_limit
        )

        # 상품 자체 LTV 기준 최대 대출액
        max_loan_product = u.house_price * (ltv_max_prod / 100.0) if pd.notna(
            ltv_max_prod) and ltv_max_prod > 0 else float('inf')

        # 최종 Max Loan 결정: 정책/법정 LTV Cap, 엄격한 정책 DSR 한도(5.0% 기준), 상품 자체 LTV 한도 중 최소값
        max_possible_loan = min(max_loan_ltv_cap, max_loan_dsr_strict, max_loan_product)

        s = score_rule(r, u, u.desired_loan_amount)

        return pd.Series({
            "score": round(s, 2),
            "final_rate_min": rmin,  # 최종 계산에 사용된 Min 금리
            "final_rate_max": rmax,  # 최종 계산에 사용된 Max 금리
            "rate_for_calc": rate_for_calc,
            "estimated_monthly_payment": calc_monthly_payment(u.desired_loan_amount, rate_for_calc / 100.0,
                                                              u.loan_term_years),
            "estimated_dsr": round(dsr, 3),
            "max_possible_loan": max_possible_loan,
        })

    metrics_df = out.apply(apply_metrics_and_score, axis=1)
    out = pd.concat([out, metrics_df], axis=1)

    out = out.sort_values("score", ascending=False)
    rc.append(ReasoningStep(step="스코어링",
                            detail=f"요청된 4가지 기준(금리, 한도, 선호유형, 기간)을 종합하여 → {len(out)}건에 점수 부여"))
    return out


def llm_explain_and_finalize(u: UserInput, top_df: pd.DataFrame, rc: List[ReasoningStep]) -> List[RecItem]:
    """LLM을 사용하여 최종 추천 항목과 그 이유를 생성하고, RAG 근거를 붙입니다."""
    if top_df.empty: return []
    head = top_df.head(5).copy()

    # _fmt_rate에 사용자 선호 금리 유형 전달
    head['rate_range_fmt'] = head.apply(lambda r: _fmt_rate(r, u.fixed_or_variable_pref), axis=1)

    top_rows_for_llm = head[[
        "bank", "product", "rate_range_fmt", "estimated_dsr", "max_possible_loan", "score"
    ]].rename(columns={'rate_range_fmt': 'rate_range'}).to_dict(orient="records")

    outs: List[RecItem] = []

    try:
        user_json = u.model_dump_json(indent=None, exclude_none=True)

        completion = client.chat.completions.create(
            # NOTE: DummyOpenAI 사용 시 실제 API 호출은 일어나지 않습니다.
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": EXPLAIN_PROMPT},
                # FIX: SyntaxError 방지를 위해 딕셔너리 구문을 {"role": "user", ...}로 명확히 수정
                {"role": "user", "content": f"[사용자]{user_json}\n[후보]{json.dumps(top_rows_for_llm, ensure_ascii=False)}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        res = completion.choices[0].message.content
        data = _safe_parse_json(res)

        for item in data.get("recommendations", [])[:3]:
            try:
                # LLM 결과 파싱
                llm_rec = RecItemLLM(**item)

                # 데이터프레임에서 계산 지표 매칭
                match = head[(head['bank'] == llm_rec.bank) & (head['product'] == llm_rec.product)]
                if not match.empty:
                    match_row = match.iloc[0]
                    outs.append(RecItem(
                        bank=llm_rec.bank, product=llm_rec.product, term_years=u.loan_term_years,
                        # 최종 계산에 사용된 금리 값을 저장
                        rate_min=match_row['final_rate_min'],
                        rate_max=match_row['final_rate_max'],
                        # --- [수정 끝] ---
                        rate_range=llm_rec.rate_range,
                        estimated_monthly_payment=match_row['estimated_monthly_payment'],
                        estimated_dsr=match_row['estimated_dsr'],
                        max_possible_loan=match_row['max_possible_loan'],
                        score=match_row['score'],
                        why=llm_rec.why,  # <--- LLM이 생성한 'why' 사용
                        documents_required = match_row['documents_required']
                    ))
            except Exception as parse_e:
                # rc.append(ReasoningStep(step="LLM 파싱 오류", detail=f"항목 파싱 실패: {parse_e.__class__.__name__}"))
                pass

    except Exception as e:
        # rc.append(ReasoningStep(step="LLM 호출 오류", detail=f"OpenAI 호출 실패: {e.__class__.__name__}"))
        pass

    # Fallback fill (LLM 호출 또는 파싱 실패 시: 상세한 규칙 기반 설명 사용)
    if len(outs) < 3:
        # rc.append(ReasoningStep(step="설명 생성 (Fallback)", detail="LLM 호출 또는 파싱 실패로 규칙 기반 설명을 생성합니다."))
        for _, r in head.iterrows():
            if len(outs) >= 3: break
            if any(o.product == r["product"] and o.bank == r["bank"] for o in outs): continue

            # 규칙 기반 'why' 문구 생성: 합리적인 설명을 제공하여 사용자의 신뢰도를 높입니다.
            rate_range_fmt = _fmt_rate(r, u.fixed_or_variable_pref)

            why_text = (
                f"이 상품은 고객님이 선호하시는 **{u.fixed_or_variable_pref}** 금리 유형을 제공합니다. "
                f"최저 금리는 **{r['final_rate_min']:.2f}%**({rate_range_fmt})로 금리 경쟁력이 우수하며, "
                f"최대 대출 가능 한도 **{r['max_possible_loan']:,.0f}원**으로 희망액({u.desired_loan_amount:,.0f}원) 확보에 유리합니다. "
                f"산정된 DSR은 **{r['estimated_dsr']:.3f}** 수준으로, 월 상환액({r['estimated_monthly_payment']:,.0f}원) 대비 상환 부담이 낮습니다."
            )

            outs.append(RecItem(
                bank=r["bank"], product=r["product"], term_years=u.loan_term_years,
                # 최종 계산에 사용된 금리 값을 저장
                rate_min=r['final_rate_min'],
                rate_max=r['final_rate_max'],
                # --- [수정 끝] ---
                rate_range=rate_range_fmt,
                estimated_monthly_payment=r['estimated_monthly_payment'],
                estimated_dsr=r['estimated_dsr'], max_possible_loan=r['max_possible_loan'],
                score=r['score'],
                why=why_text,
                documents_required=r['documents_required'] # <-- 상세한 규칙 기반 설명 사용
            ))

    # RAG 근거 추가 (수정된 retrieve_terms_for_product 함수 사용)
    for r in outs[:3]:
        # 'head' DataFrame에서 해당 상품의 row를 찾습니다.
        product_row = head[(head['bank'] == r.bank) & (head['product'] == r.product)]
        if not product_row.empty:
            evidence_text = retrieve_terms_for_product(product_row.iloc[0])
            evidence_list = [e.strip() for e in evidence_text.split('\n\n') if e.strip()]
            r.evidence = evidence_list
        else:
            r.evidence = ["상품 데이터 추출 실패."]

    return outs[:3]



def recommend_loans(df: pd.DataFrame, user: UserInput) -> RecOutput:
    """전체 대출 추천 프로세스를 실행합니다."""
    rc: List[ReasoningStep] = []

    if df.empty:
        # DB가 없는 경우 더미 데이터 생성
        df = pd.DataFrame([{
            "bank": "A은행", "product": "고정형 주담대", "loan_type": "주담대",
            "locked_min": 3.8, "locked_max": 4.9,  # 고정 금리 더미
            "adaptive_min": 4.0, "adaptive_max": 5.1,  # 변동 금리 더미
            "max_ltv": 70,
            "eligible_rule": "무주택 또는 1주택", "advantage_tips": "급여이체, 첫거래", "doc_path": "",
            "doc_text": "상품설명서: 고정 금리 3.8% ~ 4.9% 적용. 최대 LTV 70%까지 가능합니다."
        },
            {
                "bank": "B은행", "product": "변동형 주담대", "loan_type": "주담대",
                "locked_min": 4.1, "locked_max": 5.2,
                "adaptive_min": 3.5, "adaptive_max": 4.5,  # 변동 금리 더미
                "max_ltv": 60,
                "eligible_rule": "무주택 또는 1주택", "advantage_tips": "첫거래", "doc_path": "",
                "doc_text": "상품설명서: 변동 금리는 3.5%부터 시작. LTV는 60%이며, 우대 조건 적용 시 최저 금리 3.5% 적용."
            },
            {
                "bank": "C은행", "product": "생애최초론", "loan_type": "정책형",
                "locked_min": 4.1, "locked_max": 5.2,  # 고정 금리 더미
                "adaptive_min": float('nan'), "adaptive_max": float('nan'),  # 변동 금리 없음 가정
                "max_ltv": 70,
                "eligible_rule": "생애최초", "advantage_tips": "우대금리", "doc_path": "",
                "doc_text": "정책형 상품: 생애최초 주택 구매자만 신청 가능. 고정 금리 4.1%~5.2% 적용. 30년 만기까지 지원합니다."
            },
        ])
        df = _ensure_columns(df)  # 더미 데이터에도 새로운 컬럼이 적용되도록 호출

    cand = rule_filter(df, user, rc)
    scored = score_products(cand, user, rc)

    recs = llm_explain_and_finalize(user, scored, rc)

    rc.append(ReasoningStep(step="완료", detail=f" 총 {len(recs)}개 추천"))

    return RecOutput(
        user=user,
        recommendations=recs,
        reasoning_chain=rc,
    )


##############################################################################################################
# --- 5. 메인 실행 함수 (수정 없음) ---
##############################################################################################################

def run_recommender(user_input_dict: dict):
    """프론트엔드 API용 함수."""
    load_policy_corpus_from_db()

    try:
        df_products = build_df_from_sqlite(PRODUCT_DB_PATH)
    except FileNotFoundError:
        df_products = pd.DataFrame(columns=REQUIRED_COLS)

    # 1) 새로운 입력 딕셔너리를 파싱하여 UserInput 모델로 변환 (Validation 적용)

    # 입력 파라미터 파싱 및 변환
    term_years = _parse_term_years(str(user_input_dict.get('loan_term_years', '20년')))
    annual_income_val = _parse_annual_income(str(user_input_dict.get('annual_income', '0')))
    # houses_owned_val: '무주택' -> 0, '1주택' -> 1
    houses_owned_str = str(user_input_dict.get('house_owned', '무주택'))
    houses_owned_val = 0 if houses_owned_str.startswith('무') else (
        int(re.search(r'\d+', houses_owned_str).group(0)) if re.search(r'\d+', houses_owned_str) else 1)
    is_first_homebuyer_val = str(user_input_dict.get('is_first_homebuyer', '아니오')).startswith('예')
    # 금리 유형 선호도 파싱
    rate_type_str = str(user_input_dict.get('rate_type', '고정금리'))
    fixed_or_variable_pref_val: Literal["고정", "변동"] = (
        "고정" if "고정" in rate_type_str else "변동"
    )

    # [추가]: 파싱된 핵심 입력 값을 출력하는 로직 추가
    # print("--- 1. 시스템 내부 파싱 결과 (UserInput 객체 생성 전) ---")
    print(f"  - 대출 기간(년): {term_years}")
    print(f"  - 연소득(원): {annual_income_val:,.0f}")
    print(f"  - 주택 보유 수: {houses_owned_val} 주택")
    print(f"  - 생애최초 여부(bool): {'True' if is_first_homebuyer_val else 'False'}")
    print(f"  - 선호 금리 유형: {fixed_or_variable_pref_val}")
    print("-" * 55)
    # [추가 끝]

    # UserInput 모델 생성 (내부 로직 유지를 위한 default 값 사용)
    user = UserInput(
        house_price=float(user_input_dict.get("house_price", 0)),
        desired_loan_amount=float(user_input_dict.get("desired_loan_amount", 0)),
        loan_term_years=term_years,
        annual_income=annual_income_val,
        houses_owned=houses_owned_val,
        is_first_homebuyer=is_first_homebuyer_val,
        property_type=user_input_dict.get("property_type", "아파트"),
        house_address=user_input_dict.get("house_address", "서울시 강남구"),
        fixed_or_variable_pref=fixed_or_variable_pref_val,  # 값 전달
        # purpose, employment_type, credit_tier는 UserInput 모델의 default 값 사용
    )

    # 2) 추천 로직 실행
    result = recommend_loans(df_products, user)

    # 3) Pydantic V2 model_dump() 사용
    return result.model_dump(exclude_none=True)


##############################################################################################################

chat_llm = ChatOpenAI(model_name="gpt-4o-mini", # gpt-3.5-turbo
                 temperature=0)  # Modify model_name if you have access to GPT-4


# 최근 대화가 max_token_limit을 초과할 경우에 요약하여 저장
memory = ConversationSummaryBufferMemory(
    llm = chat_llm, # 대화를 요약하며 메모리 저장하기 위한 기능의 llm 설정
    max_token_limit = 7000, # 최근대화 토큰 수가 80을 넘어가면 요약, 2000, 5000
    memory_key = 'chat_history', # 메모리에 "chat_history"라는 키로 대화를 저장, Default 값은 history
    return_messages = True #  True로 설정 시 messages의 리스트형태로 histroy 항목을 반환, False로 설정 시 string으로 반환, Default 값은 False
)

def load_memory(input):
    print( input )
    return memory.load_memory_variables({})['chat_history'] # 메모리에 "chat_history"라는 키로 저장된 대화기록 반환

system_prompt = (
        "당신은 주택담보대출 상품 추천 및 상담 전문가입니다. "
        "사용자의 질문에 응답할 때, 도구(run_recommender)를 호출할지 결정합니다. "
        "도구 호출에 필요한 모든 정보가 있을 때만 도구를 호출해야 합니다. "
        "만약 이전에 추천 결과가 메모리에 있다면, 도구를 다시 호출하지 않고 메모리 내용을 참조하여 답변하세요."
        "항상 가장 최근 도구 호출 결과 한 번만 기준으로 답하라"
        "가장 최근 결과가 ‘추천 없음’일 경우, 과거에 있었던 추천 목록은 사용하지 말라"
        "** 대신 [] 이걸로 출력해줘  "
    )

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human','{question}')
    ]
)


def invoke_chain(question: Optional[str] = None, tool_output: Optional[List[Dict[str, Any]]] = None):
    """
    도구 호출 결과(tool_output)를 메모리에 저장하거나,
    질문(question)을 처리하고 결과를 메모리에 저장합니다.
    """
    # ------------------------------
    # 1. 추천 결과가 0개거나 product가 0인 경우 → 메모리 초기화
    # ------------------------------

    print("tool_output:", tool_output)
    if tool_output is None or len(tool_output) == 0:
    # if not rec.get("product"):
        print("test")
        if question:
            print("상품명이 0 또는 None → 메모리 초기화 수행")
            # memory.chat_memory.messages = memory.chat_memory.messages[-2:]
            memory.chat_memory.messages.clear()
            # memory.clear()
            result = chain.invoke({'question': question})

            # memory.save_context(
            #     {'input': question},
            #     {'output': result.content}

            # )

            return result
    # 추천 결과 중 product가 0인 경우
    #     for rec in tool_output:
    #     if not rec.get("product"):  # product가 0 또는 None 또는 '' 인 경우
        # memory.clear()
        # return None

    # ------------------------------
    # 2. tool_output 정상 저장 로직
    # ------------------------------
    if tool_output:
        summary_list = []
        for rec in tool_output:
            summary_list.append({
                "은행": rec['bank'],
                "상품": rec['product'],
                "금리범위": rec.get('rate_range', '정보 없음'),
                "최대대출가능액": f"{rec['max_possible_loan']:,.0f}원",
                "추천이유": rec['why']
            })

        summary_json = json.dumps(summary_list, indent=2, ensure_ascii=False)

        input_message = '주택담보대출 추천해줘'
        output_message = f"추천 상품 목록 (참조): {summary_json}"

        memory.save_context(
            {'input': input_message},
            {'output': output_message}
        )

    # ------------------------------
    # 3. 질문 처리 로직
    # ------------------------------
    if question:
        result = chain.invoke({'question': question})

        memory.save_context(
            {'input': question},
            {'output': result.content}

        )
        return result

    # return None
    # 1. tool_output 저장 로직 (질문 유무와 상관없이 실행 가능)
    # if tool_output:
    #     # 핵심 정보 추출 및 저장 로직 (이전 답변의 2번 개선 사항 적용)
    #     summary_list = []
    #     for rec in tool_output:
    #         # LLM이 쉽게 참조할 수 있도록 핵심 정보만 추출
    #         summary_list.append({
    #             "은행": rec['bank'],
    #             "상품": rec['product'], #
    #             "금리범위": rec.get('rate_range', '정보 없음'),
    #             "최대대출가능액": f"{rec['max_possible_loan']:,.0f}원",
    #             "추천이유": rec['why']
    #         })
    #
    #     summary_json = json.dumps(summary_list, indent=2, ensure_ascii=False)
    #
    #     # 메모리에 추천 결과를 AI 응답 형태로 저장 (LLM이 참고하도록 문맥 제공)
    #     # LLM에게 전달할 인풋 메시지(가상의 사용자 요청)
    #     input_message = '주택담보대출 추천해줘'
    #     # 메모리에 저장할 아웃풋 메시지(추천 결과)
    #     output_message = f"추천 상품 목록 (참조): {summary_json}"
    #
    #     # 여기서 메모리에 저장합니다.
    #     memory.save_context(
    #         {'input': input_message},
    #         {'output': output_message}
    #     )
    #
    # # 2. 질문 처리 로직 (question이 있을 때만 실행)
    # if question:
    #     # 이제, 기존 체인 호출 로직을 사용
    #     # load_memory를 통해 이미 tool_output 저장 내역이 chat_history에 포함됨
    #     result = chain.invoke({'question': question})
    #
    #     # 마지막 질문과 응답을 저장
    #     memory.save_context(
    #         {'input': question},
    #         {'output': result.content}
    #     )
    #     return result
    #
    # # 질문이 없고 tool_output만 전달된 경우
    # return None

chain = RunnablePassthrough.assign(chat_history=load_memory) | prompt |  chat_llm


##############################################################################################################

def main():
    """DB 추출부터 추천까지의 전체 프로세스를 실행합니다."""
    print("==================================================")

    # 1. 정책 코퍼스 로드
    load_policy_corpus_from_db()

    # 2. 상품 데이터 로드
    print(f"1. SQLite DB ({PRODUCT_DB_PATH})에서 상품 약관 추출 및 DataFrame 생성...")
    try:
        df_products = build_df_from_sqlite(PRODUCT_DB_PATH)
        print(f"[OK] DB에서 총 {len(df_products)}건 상품 추출 완료")
    except FileNotFoundError as e:
        print(f"[ERROR] 상품 DB 로드 실패: {e}. 더미 데이터로 대체합니다.")
        df_products = pd.DataFrame(columns=REQUIRED_COLS)

    print("--------------------------------------------------")

    # 3. 사용자 입력 (JSON 출력에 맞게 수정)
    new_user_input = {
        'house_address': '광주광역시',
        'property_type': '아파트',
        'house_price': 500_000_000,  ## 주택 가격
        'desired_loan_amount': 200_000_000,  ## 대출 희망 금액
        'annual_income': '10000~15000',
        'loan_term_years': '30년',
        'house_owned': '무주택',  # 무주택 # 1주택
        'is_first_homebuyer': '예',  # 예 # 아니오
        'fixed_or_variable_pref': '변동',  # 고정 # 변동
    }

    # 사용자 입력 정리 및 출력
    print("2. 사용자 입력 데이터:")
    print(f"  - 주택 주소/유형: {new_user_input.get('house_address')}, {new_user_input.get('property_type')}")
    print(f"  - 주택 가격: {new_user_input.get('house_price', 0):,.0f}원")
    print(f"  - 희망 대출액: {new_user_input.get('desired_loan_amount', 0):,.0f}원")
    print(f"  - 연소득 정보: {new_user_input.get('annual_income')}")
    print(f"  - 대출 기간: {new_user_input.get('loan_term_years')}")
    print(f"  - 주택 소유 여부: {new_user_input.get('house_owned')}")
    print(f"  - 생애 최초 구매: {new_user_input.get('is_first_homebuyer')}")
    print(f"  - 선호 금리 유형: {new_user_input.get('fixed_or_variable_pref')}")
    print("--------------------------------------------------")

    # 4. 대출 추천 시스템 실행
    print(
        f"3. 대출 추천 시스템 실행 (주소: {new_user_input['house_address']}, 희망액:{new_user_input['desired_loan_amount']:,.0f}원)...")
    try:
        # run_recommender 함수 호출 시 정책/한도 계산 로직이 실행됨
        result_dict = run_recommender(new_user_input)
        recommendations = result_dict.get('recommendations', [])

        print("[OK] 추천 완료")
        print("--------------------------------------------------")
        print("4. 최종 추천 결과:")

        # 정책 한도 계산 결과 출력
        ltv_step = next((s for s in result_dict['reasoning_chain'] if s['step'] == '정책 LTV 한도'), None)
        if ltv_step:
            print(f"**정책 LTV 한도 결정**: {ltv_step['detail']}")

        # 정책 DSR 한도 계산 결과 출력 (새로 추가)
        dsr_step = next((s for s in result_dict['reasoning_chain'] if s['step'] == '정책 DSR 한도'), None)
        if dsr_step:
            print(f"**정책 DSR 한도 결정**: {dsr_step['detail']}")

        # 추천 상품의 최대 가능 대출 금액 출력
        for i, rec in enumerate(result_dict['recommendations']):
            print(f"[{i + 1}] {rec['bank']} {rec['product']} - 최대 가능 금액: {rec['max_possible_loan']:,.0f}원")

        # 전체 JSON 출력
        print("\n=== 전체 JSON 출력 ===")
        print(json.dumps(result_dict, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"[ERROR] 사용자 입력/추천 시스템 실행 실패: {e.__class__.__name__}: {e}")
        sys.exit(1)

    print("==================================================")

    print("\n" + "=" * 80)
    print("[시스템] 첫 질문 전에 추천 결과를 메모리에 저장 중...")
    invoke_chain(tool_output=recommendations)
    print("[시스템] 메모리 저장 완료.")
    print(memory.load_memory_variables({}))  # 메모리 저장 내용 확인

    print("\n" + "=" * 80)
    reponse = invoke_chain(
        question='방금 추천해준 상품 중 첫번째 상품의 금리 범위와 추천 이유는 뭐였지?'
    )
    # reponse = invoke_chain(
    #     question='추천 상품이 없는 이유를 알려줘?'
    # )
    print(reponse.content)
    print(memory.load_memory_variables({}))

    print("\n" + "=" * 80)
    reponse = invoke_chain('방금 추천해준 상품 중 두번째 상품의 금리 범위와 추천 이유는 뭐였지?')
    print(reponse.content)
    print(memory.load_memory_variables({}))

    print("\n" + "=" * 80)
    reponse = invoke_chain('방금 추천해준 상품 중 세번째 상품의 금리 범위와 추천 이유는 뭐였지?')
    print(reponse.content)
    print(memory.load_memory_variables({}))

    print("\n" + "=" * 80)
    reponse = invoke_chain('조금전 이야기한 상품에 대한 차이를 비교해줘')
    print(reponse.content)
    print(memory.load_memory_variables({}))

if __name__ == "__main__":
    main()
