from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
import math

from rec_product_logic_conv_F import UserInput, run_recommender, RecOutput

from fastapi.encoders import jsonable_encoder
from rec_product_logic_conv_F import invoke_chain

app = FastAPI()

# CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# templates / static 연결
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


# ---------------------------
# 사용자 입력 모델
# ---------------------------
class Page1Data(BaseModel):
    loanAmount: int
    repaymentPeriod: str
    rateType: str
    # HTML에서 보내는 이름 그대로 유지

class Page2Data(BaseModel):
    housePrice: int
    houseAddress: str
    collateralType: str

class Page3Data(BaseModel):
    income: str
    housingCount: str
    firstHome: str

class ChatRequest(BaseModel):
    message: str


# ---------------------------
# 입력 저장용 전역 dict
# ---------------------------
user_inputs = {}

# ---------------------------
# 결괏값저장 dict
# ---------------------------
recommender_cache = []

# ---------------------------
# 페이지 렌더링
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    await reset_user_input()
    return templates.TemplateResponse("banner.html", {"request": request})

@app.get("/page1", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/page2", response_class=HTMLResponse)
async def page2(request: Request):
    return templates.TemplateResponse("page2.html", {"request": request})

@app.get("/page3", response_class=HTMLResponse)
async def page3(request: Request):
    return templates.TemplateResponse("page3.html", {"request": request})

@app.get("/loan_analysis", response_class=HTMLResponse)
def analysis_page(request: Request):
    return templates.TemplateResponse(
        "loan_analysis.html",
        {"request": request}
    )

@app.get("/api/recommendations")
def api_recommendations():
    recommender_cache, rec_result = compute_recommendations()

    # ---- 여기부터 신규 로직 ----
    # 추천 결과가 있을 때만 LangChain 메모리에 저장
    recommendations = []
    if rec_result is not None:
        recommendations = rec_result.get("recommendations", [])

    if recommendations:  # 0개가 아니면 = 뭔가 있을 때만
        invoke_chain(tool_output=recommendations)
    else:
        # 추천이 0개일 때는 과거 추천 결과를 덮어쓰지도, 새로 쓰지도 않음
        # (더 깔끔하게 하려면 rec_product_logic_conv_F 쪽에서
        #  "현재 조건에서는 추천 불가" 같은 문장을 메모리에 저장하는 함수 하나 만드는 것도 좋음)
        pass
    # ---- 신규 로직 끝 ----

    safe_result = jsonable_encoder(rec_result)
    # NaN/Infinity 방어
    for rec in safe_result.get("recommendations", []):
        for key in ["rate_min", "rate_max", "max_possible_loan"]:
            value = rec.get(key, 0)
            if (
                value is None
                or not isinstance(value, (int, float))
                or math.isnan(value)
                or math.isinf(value)
            ):
                rec[key] = 0  # 안전하게 0으로 치환
    return JSONResponse(content=safe_result)


# @app.get("/api/recommendations")
# def api_recommendations():
#     rec_result = compute_recommendations()
#     invoke_chain(tool_output=rec_result["recommendations"])
#     safe_result = jsonable_encoder(rec_result)
#     # NaN/Infinity 방어
#     for rec in safe_result.get("recommendations", []):
#         for key in ["rate_min", "rate_max", "max_possible_loan"]:
#             value = rec.get(key, 0)
#             if value is None or not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
#                 rec[key] = 0  # 안전하게 0으로 치
#     return JSONResponse(content=safe_result)

@app.get("/loan_products", response_class=HTMLResponse)
def loan_products_page(request: Request):
    return templates.TemplateResponse(
        "loan_products.html",
        {"request": request}
    )

@app.get("/product_detail", response_class=HTMLResponse)
async def product_detail(request: Request):
    return templates.TemplateResponse(
        "product_detail.html",
        {"request": request}
    )


# @app.get("/result", response_class=HTMLResponse)
# async def result(request: Request):
#     user_data = load_user_input()
#     user_obj = json_to_user_input(user_data)
#     rec_result: RecOutput = recommend_loans("load_product_terms.csv", user_obj)
#
#     # 페이지 1~4 데이터를 통합
#     return templates.TemplateResponse("result.html", {"request": request, "result": rec_result,  # 추천 결과
#             "user_input": user_data})


# ---------------------------
# API (각 페이지 POST)
# ---------------------------
@app.post("/api/page1")
async def api_page1(data: Page1Data):
    # user_inputs['page1'] = data.dict()
    # print("Page1 데이터:", data)  # 서버 콘솔 확인
    user_inputs["desired_loan_amount"] = data.loanAmount
    user_inputs["loan_term_years"] = data.repaymentPeriod
    user_inputs["rate_type"] = data.rateType
    print("현재 누적 데이터:", user_inputs)
    save_json()
    return JSONResponse({"status": "ok"})

@app.post("/api/page2")
async def api_page2(data: Page2Data):
    # user_inputs['page2'] = data.dict()
    # print("Page2 데이터:", data)  # 서버 콘솔 확인
    user_inputs["house_price"] = data.housePrice
    user_inputs["house_address"] = data.houseAddress
    user_inputs["property_type"] = data.collateralType
    print("현재 누적 데이터:", user_inputs)
    save_json()
    return JSONResponse({"status": "ok"})

@app.post("/api/page3")
async def api_page3(data: Page3Data):
    # user_inputs['page3'] = data.dict()
    # print("Page3 데이터:", data)
    user_inputs["annual_income"] = data.income
    user_inputs["house_owned"] = data.housingCount
    user_inputs["is_first_homebuyer"] = data.firstHome
    print("현재 누적 데이터:", user_inputs)
    save_json()
    print(user_inputs)
    return JSONResponse({"status": "ok"})

@app.post("/api/chat")
async def chat(req: ChatRequest):
    user_msg = req.message

    tool_output = recommender_cache if isinstance(recommender_cache, list) else None
    # LangGraph 체인으로 질문을 보냄
    result = invoke_chain(question=user_msg, tool_output=tool_output)

    return {"reply": result.content}


def save_json():
    with open("user_input.json", "w", encoding="utf-8") as f:
        json.dump(user_inputs, f, ensure_ascii=False, indent=2)

def load_user_input() -> dict:
    try:
        with open("user_input.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def json_to_user_input(data: dict) -> UserInput:
    return UserInput(
        purpose=data.get("purpose", "기타"),
        annual_income=int(data.get("monthlyIncome", 0)),
        employment=data.get("employment", "기타"),
        houses_owned={"무주택": 0, "1주택": 1, "다주택": 2}.get(data.get("housingStatus"), 0)

    )




def compute_recommendations():
    global recommender_cache
    user_data = load_user_input()

    # dict 비교를 JSON 문자열로 변경 (값 기반 비교)
    # prev_input_json = json.dumps(recommender_cache.get("input"), sort_keys=True)
    # current_input_json = json.dumps(user_data, sort_keys=True)

    # if prev_input_json != current_input_json:
    rec_result = run_recommender(user_data)
    print("사용자 입력이 변경되어 추천 재계산")
    recommender_cache = rec_result.get("recommendations", [])
    # recommender_cache["input"] = user_data
    # else:

    print(rec_result.get("recommendations", []))

    return recommender_cache, rec_result

# # ----------------------------
# # (선택) user_input.json 리셋 API
# # ----------------------------
# @app.get("/reset-user-input")
# async def reset_user_input():
#     """
#     user_input.json을 빈 객체로 초기화하는 엔드포인트 (선택)
#     """
#     try:
#         with open("user_input.json", "w", encoding="utf-8") as f:
#             f.write("{}")
#         return {"status": "reset complete"}
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"error": str(e)},
#         )

@app.get("/reset-user-input")
async def reset_user_input():
    """
    user_input.json + 서버 메모리 캐시(user_inputs, recommender_cache) 초기화
    """
    global user_inputs, recommender_cache
    try:
        # 1) 서버 메모리(dict) 초기화
        user_inputs.clear()
        recommender_cache = []

        # 2) 파일 초기화
        with open("user_input.json", "w", encoding="utf-8") as f:
            f.write("{}")

        # 3) (선택) LangChain 메모리까지 비우고 싶으면 rec_product_logic_conv_F에
        #    clear_memory() 같은 함수 하나 만들어서 여기서 호출하면 됨.
        # from rec_product_logic_conv_F import clear_memory
        # clear_memory()

        return {"status": "reset complete"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

