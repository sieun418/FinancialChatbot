const messageArea = document.querySelector(".message-area");
const input = document.querySelector(".chat-input input");
const sendBtn = document.querySelector(".chat-input button");
const completeBtn = document.querySelector(".complete-btn");
const pulseDots = document.querySelector(".pulse-dots");

let currentStep = 0;
let analysisSteps = [];
let analysisData = null;

/* ===============================
    메시지 출력 (ID 부여)
================================= */
function appendMessage(sender, text) {
  const bubble = document.createElement("div");
  const id = "msg_" + Date.now() + "_" + Math.random().toString(36).slice(2);

  bubble.id = id;
  bubble.className = `message-bubble ${sender}`;
  bubble.textContent = text;

  messageArea.appendChild(bubble);
  messageArea.scrollTop = messageArea.scrollHeight;

  return id;
}

/* 메시지 제거 */
function removeMessage(id) {
  if (!id) return;
  const el = document.getElementById(id);
  if (el) el.remove();
}

/* ===============================
    JSON 분석 데이터 로드
================================= */
fetch("/api/recommendations")
  .then(res => res.json())
  .then(data => {
    analysisData = data;

    analysisSteps = data.reasoning_chain.map(item => ({
      step: item.step,
      text: item.detail
    }));

    analysisSteps.push({
      step: "완료",
      text: "분석이 완료되었습니다!"
    });

    runAnalysisStep();
  })
  .catch(err => {
    console.error("JSON 불러오기 실패:", err);
    appendMessage("bot", "데이터를 불러올 수 없습니다.");
  });

/* ===============================
    분석 단계 자동 표시
================================= */
function runAnalysisStep() {
  if (!analysisSteps.length) return;

  if (currentStep >= analysisSteps.length) {
    document.getElementById("header-icons").classList.add("hidden");
    completeBtn.classList.remove("hidden");
    appendMessage("bot", "🎉 분석이 끝났습니다! 추천 상품을 확인해보세요.");
    return;
  }

  const step = analysisSteps[currentStep];
  appendMessage("bot", `🔎 [${step.step}]\n${step.text}`);

  currentStep++;

  setTimeout(runAnalysisStep, 1000);
}

/* ===============================
    LangGraph로 메시지 전송
================================= */
sendBtn.addEventListener("click", async () => {
  const text = input.value.trim();
  if (!text) return;

  appendMessage("user", text);
  input.value = "";

  // 로딩 메시지 출력
  const loadingId = appendMessage("bot", "답변을 불러오는 중입니다...");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    });

    const data = await res.json();

    removeMessage(loadingId);

    // LangGraph 응답 구조: { reply: "응답 내용" }
    appendMessage("bot", data.reply ?? "응답을 이해하지 못했어요.");
  } catch (err) {
    console.error(err);
    removeMessage(loadingId);
    appendMessage("bot", "서버 연결 오류가 발생했어요!");
  }
});

/* 엔터키 */
input.addEventListener("keypress", e => {
  if (e.key === "Enter") sendBtn.click();
});

/* 다음 페이지 이동 */
completeBtn.addEventListener("click", () => {
  localStorage.setItem("loan_result", JSON.stringify(analysisData));
  window.location.href = "/loan_products";
});
