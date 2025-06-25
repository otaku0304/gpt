document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('user-input').addEventListener('keypress', function(e) {
  if (e.key === 'Enter') sendMessage();
});

function appendMessage(text, sender) {
  const messageDiv = document.createElement('div');
  messageDiv.classList.add('message', sender);
  messageDiv.textContent = text;
  document.getElementById('chat-box').appendChild(messageDiv);
  document.getElementById('chat-box').scrollTop = document.getElementById('chat-box').scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById('user-input');
  const userText = input.value.trim();
  if (!userText) return;

  appendMessage(userText, 'user');
  input.value = '';

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userText })
    });

    const data = await response.json();
    appendMessage(data.reply || 'No response received.', 'bot');
  } catch (error) {
    appendMessage('Error: Could not reach server.', 'bot');
  }
}
