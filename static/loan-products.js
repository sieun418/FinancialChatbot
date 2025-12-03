// DOM 로드 후 실행
document.addEventListener("DOMContentLoaded", async () => {

  // JSON 불러오기
  const rawData = JSON.parse(localStorage.getItem("loan_result") || "{}");
  const rawProducts = rawData.recommendations || [];

  // 방어 코드: 데이터가 없으면 종료
  if (!rawProducts.length) {
    console.error("추천 상품 데이터가 없습니다:", rawData);
    document.getElementById("productCount").innerText = "추천 상품이 없습니다.";
    return;
  }

  // JSON 구조 변환
  const products = rawProducts.map(r => ({
    bankName: r.bank,
    productName: r.product,
    termYears: r.term_years,
    minRate: r.rate_min,
    maxRate: r.rate_max,
    estimatedMonthlyPayment: r.estimated_monthly_payment,
    estimatedDSR: r.estimated_dsr,
    maxLimit: r.max_possible_loan,
    score: r.score,
    description: r.why,
    requiredDocuments: r.documents_required, // 필수서류 포함
    recommended: true
  }));

  // 상품 개수 표시
  document.getElementById("productCount").innerText =
    `고객님께 딱 맞는 상품 ${products.length}개를 찾았어요`;

  const list = document.getElementById("productList");

  // 카드 생성
  products.forEach((r, idx) => {
    // 카드 클릭 시 저장할 객체
    const cardData = {
      bankName: r.bankName,
      productName: r.productName,
      termYears: r.termYears,
      minRate: r.minRate,
      maxRate: r.maxRate,
      estimatedMonthlyPayment: r.estimatedMonthlyPayment,
      estimatedDSR: r.estimatedDSR,
      maxLimit: r.maxLimit,
      score: r.score,
      description: r.description,
      requiredDocuments: r.requiredDocuments, // 필수서류 확실히 포함
      recommended: r.recommended
    };

    const card = document.createElement("div");
    card.className = `product-card fade-up ${r.recommended ? "recommend" : ""}`;
    card.style.animationDelay = `${idx * 0.1}s`;

    card.innerHTML = `
      <div class="bank-row">
        <div class="bank-name">${r.bankName}</div>
        ${r.recommended ? `<span class="badge">추천</span>` : ""}
      </div>

      <div class="product-name">${r.productName}</div>

      <div class="info-row">
        <div class="rate-section">
          <p class="label">예상 금리</p>
          <p class="rate-value">${r.minRate}% ~ ${r.maxRate}%</p>
        </div>

        <div class="limit-section">
          <p class="label">최대 한도</p>
          <p class="limit-value">${(r.maxLimit / 100000000).toFixed(0)}억</p>
        </div>
      </div>
    `;

    // 클릭 시 필수서류 포함한 cardData를 저장
    card.addEventListener("click", () => {
      localStorage.setItem("selected_product", JSON.stringify(cardData));
      window.location.href = "/product_detail";
    });

    list.appendChild(card);
  });
});
