// 챗봇 열기/닫기
const chatBtn = document.getElementById("chatBtn");
const chatPanel = document.getElementById("chatPanel");
const closeChat = document.getElementById("closeChat");

chatBtn.addEventListener("click", () => {
  chatPanel.style.width = "400px";
  chatBtn.style.display = "none";

  const hint = document.querySelector(".help-hint");
  if (hint) {
    hint.style.opacity = "0";
    setTimeout(() => hint.remove(), 300);
    }
});

closeChat.addEventListener("click", () => {
  chatPanel.style.width = "0";
  chatBtn.style.display = "block";
});


// 메세지 전송
const chatInput = document.getElementById("chatInput");
const chatBody = document.getElementById("chatBody");
const sendChat = document.getElementById("sendChat");

function addMessage(message, sender = "user") {
  const msg = document.createElement("div");
  msg.classList.add("msg", sender);
  msg.innerText = message;

  chatBody.appendChild(msg);
  chatBody.scrollTop = chatBody.scrollHeight;
}


// 서버로 메시지 보내고 화면에 추가
async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;

  addMessage(text, "user");
  chatInput.value = "";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    });

    const data = await response.json();
    addMessage(data.reply, "ai");
  } catch (error) {
    console.error("서버 전송 실패:", error);
    addMessage("서버 오류가 발생했습니다.", "ai");
  }
}

// 전송 버튼 클릭
sendChat.addEventListener("click", sendMessage);

// 엔터키 전송
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    sendMessage();
  }
});


// 안내문구 자동소멸
setTimeout(() => {
  const hint = document.querySelector(".help-hint");
  if (hint) {
    hint.style.opacity = "0";
    setTimeout(() => hint.remove(), 500); // 완전히 사라지면 DOM에서 삭제
  }
}, 9000);