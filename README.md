# 🏠 FinancialChatbot
### AI 기반 주택담보대출 상품 추천 서비스
**FastAPI · LangChain/LangGraph · SQLite**

---

## 📌 Overview
**MortgageAI**는 사용자가 입력한 재무·부동산 정보를 분석하여,  
은행별 **주택담보대출 상품을 추천**하고 **LTV·DTI·DSR 정책을 기반으로 대출 가능액을 분석**하는 대화형 금융 서비스입니다.

본 프로젝트는 KDT AI Academy 과정에서 수행되었으며,  
실제 금융 정책과 상품 구조를 반영하여 **현실성 있는 추천 알고리즘**을 구현하는 것을 목표로 합니다.

---

## ✨ Features

### ⭐ 1. 대출 상품 자동 추천 엔진
- 사용자 입력 기반(담보가액, 필요 대출금, 소득, 부채 등)
- SQLite에 저장된 금융상품 DB 기반 추천
- 금리, 한도, 조건 등 비교 후 Top-N 상품 제시

### ⭐ 2. 금융정책 계산 엔진 (LTV · DTI · DSR)
- 금융감독원 기준 반영
- 규제지역/비규제지역 정책 차등 적용
- 생애최초, 무주택, 1주택, 신혼부부 등 특성 반영
- 정책 계산 결과가 실제 추천 로직에 직접 사용됨

### ⭐ 3. LangChain & LangGraph 기반 Reasoning
- 다단계 추론 구조(Multi-step Reasoning)
- 상품 비교·설명 및 금융 용어 해석 자동 생성
- 사용자 조건에 맞춘 자연어 기반 금융 가이드 제공

### ⭐ 4. FastAPI + Jinja2 기반 웹 UI
- `/page1 → page2 → result` 흐름의 사용자 입력 단계 구성
- 서버 내부에서 정책 계산 → 추천 → 결과 시각화  
- REST API 없이 템플릿 렌더링 방식으로 서비스 운영

---

## 🏗️ System Architecture

