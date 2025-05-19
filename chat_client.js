const fetch = require('node-fetch');

async function sendMessage(speakerId, targetId, botRole, userInput) {
  const API_URL = 'http://localhost:8000/send_message';
  const API_KEY = 'your-secure-api-key';

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': API_KEY
      },
      body: JSON.stringify({
        speaker_id: speakerId,
        target_id: targetId,
        bot_role: botRole,
        user_input: userInput
      })
    });

    const data = await response.json();
    if (data.error) {
      console.error('Error:', data.error);
    } else {
      console.log('Response:', data.response);
    }
  } catch (error) {
    console.error('Request failed:', error.message);
  }
}

// Example usage
sendMessage('user2', 'user1', 'sister', 'what else type of movie you like?');