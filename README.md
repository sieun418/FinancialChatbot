🏠 FinancialChatbot
AI 기반 주택담보대출 상품 추천 & 정책 분석 서비스

FastAPI · LangChain/LangGraph · SQLite · 금융 정책 계산 엔진

📌 Overview

MortgageAI는 사용자의 재무 상황과 주택 정보를 입력받아,
은행별 주택담보대출 상품을 자동 추천하고 LTV·DTI·DSR 정책을 기반으로 대출 가능액을 계산하는 AI 금융 서비스입니다.

본 프로젝트는 KDT AI Academy 과정에서 진행되었으며,
실제 금융 서비스에서 요구되는 정책 로직과 사용자 경험을 반영하는 것을 목표로 설계되었습니다.

✨ Features
⭐ 1. 대출 상품 자동 추천

담보가액, 필요 대출금, 소득, 부채, 주택 수 등 입력 기반

금융상품 DB(SQLite) 기반 추천

금리/특약/상품 구조 비교 분석

⭐ 2. 금융정책 엔진 (LTV · DTI · DSR)

금융감독원 기준 반영

규제지역 여부, 생애최초/신혼부부/무주택 조건 적용

정책 산출값이 최종 대출 가능액에 반영됨

⭐ 3. LangChain & LangGraph 기반 Reasoning

다단계 판단(Multi-step Reasoning)

상품 비교·설명 자동 생성

사용자의 조건에 따라 자연어 분석 결과 제공

⭐ 4. FastAPI 기반 UI/REST API

/page1 → page2 → result 입력 흐름 제공

Jinja2 템플릿 기반 웹 UI

JSON API 응답 형태로도 결과 제공 가능

🏗️ System Architecture
User
  ↓
FastAPI (UI & API)
  ↓
Input Parsing Layer
  ↓
Policy Engine (LTV / DTI / DSR)
  ↓
Product DB Filter & Matching
  ↓
LangGraph Reasoning Agent
  ↓
Final Recommendation
  ↓
UI / JSON Response

📁 Project Structure
project/
│
├── server.py                     # FastAPI 서버 메인 엔트리
├── rec_product_logic_conv_F.py   # 정책 계산 + 추천 엔진 핵심 모듈
│
├── database/
│   ├── product.db                # 금융상품 DB (SQLite)
│   ├── policy_rules.db           # 정책 규정 DB
│
├── templates/
│   ├── page1.html
│   ├── page2.html
│   ├── result.html
│
├── static/
│   ├── css/
│   └── js/
│
└── README.md

🚀 Installation
1. Clone repo
git clone <your-repo-url>
cd <repo-folder>

2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

▶️ Run Server
uvicorn server:app --reload --port 8000


Open in browser:
👉 http://127.0.0.1:8000/page1

🧮 Policy Calculation Examples
✔ LTV (Loan-to-Value)
조건	적용 LTV
생애최초	최대 80%
무주택/1주택	70%
투기지역	40%
조정대상지역	50%
비규제지역	70%
✔ DSR (Debt Service Ratio)
DSR = (총부채 원리금 상환액 / 연소득) × 100
은행권 기본 DSR 규제 40%

✔ Final Loan Limit
대출 가능 금액 = min( LTV 기반 가능액 , DSR 기반 가능액 )

🤖 LangGraph Reasoning Flow

사용자 입력을 조건별로 파싱

규정 위반/결격 여부 확인

정책 엔진 계산

DB에서 상품 후보 생성

LangGraph가 상품 비교 및 설명 생성

Top-N 추천 결과 반환

🔧 Tech Stack

Backend: FastAPI
AI: LangChain, LangGraph, OpenAI API
DB: SQLite (상품·정책 DB)
Frontend: Jinja2 Templates
DevOps: Uvicorn, Python venv

🛠️ Future Improvements

은행 우대금리 조건 자동 계산

월 상환액 시뮬레이터 추가

더 세분화된 정책 DB 구축

대출 시나리오 기반 “맞춤형 금융 코치" 기능 확장

RAG 문서 정확도 개선

📄 License

This project is for educational & research purpose only.
본 서비스는 실제 금융 자문을 제공하지 않으며, 금융거래 결정은 반드시 공인 금융기관의 안내를 확인해야 합니다.
