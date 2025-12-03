// 로컬 스토리지에서 선택된 상품 불러오기
const selected = JSON.parse(localStorage.getItem("selected_product") || "{}");
const mainCard = document.getElementById("mainCard");
const descriptionCard = document.getElementById("descriptionCard");
const documentCard = document.getElementById("documentCard");

// 처음에는 애니메이션용 클래스 추가
mainCard.classList.add("fade-up");
descriptionCard.classList.add("fade-up");
documentCard.classList.add("fade-up");

// DOM 요소
const productName = document.getElementById("productName");
const bankName = document.getElementById("bankName");
const rateRange = document.getElementById("rateRange");
const monthlyPayment = document.getElementById("monthlyPayment");
const loanLimit = document.getElementById("loanLimit");
const productDescription = document.getElementById("productDescription");
const requiredDocuments = document.getElementById("required-documents");

// 버튼
const backBtn = document.getElementById("backBtn");
const otherProductsBtn = document.getElementById("otherProductsBtn");
const homeBtn = document.getElementById("homeBtn");

// 버튼 이벤트
backBtn.addEventListener("click", () => history.back());
otherProductsBtn.addEventListener("click", () => window.location.href = "/loan_products");
homeBtn.addEventListener("click", () => {
localStorage.removeItem("loan_result");
window.location.href = "/"});

// 상품 정보 표시
if (selected) {
  productName.innerText = selected.productName || "상품명 없음";
  bankName.innerText = selected.bankName || "은행명 없음";
  rateRange.innerText = selected.minRate !== undefined && selected.maxRate !== undefined
    ? `${selected.minRate}% ~ ${selected.maxRate}%`
    : "금리 정보 없음";
  monthlyPayment.innerText = selected.estimatedMonthlyPayment
    ? Math.round(selected.estimatedMonthlyPayment).toLocaleString() + "원"
    : "0원";
  loanLimit.innerText = selected.maxLimit
    ? Math.round(selected.maxLimit).toLocaleString() + "원"
    : "정보 없음";
  productDescription.innerText = selected.description || "상품 설명이 없습니다.";
  let rawDocs = selected.requiredDocuments || "필요 서류 정보가 없습니다.";
  let preProcessedDocs = rawDocs.split(/\d+\./)[0].trim();
  requiredDocuments.innerHTML = preProcessedDocs.replace(/\n/g, "<br>"); // 줄바꿈 유지


  setTimeout(() => mainCard.classList.add("show"), 100);   // 메인 카드
  setTimeout(() => descriptionCard.classList.add("show"), 400);
  setTimeout(() => documentCard.classList.add("show"), 700);// 설명 카드
}