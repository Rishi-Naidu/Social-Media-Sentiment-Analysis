import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [text, setText] = useState('');
  const [sentiment, setSentiment] = useState('');

  const handleSubmit = async () => {
    const response = await axios.post('http://localhost:5000/predict', { text });
    setSentiment(response.data.sentiment);
  };

  return (
    <div>
      <textarea value={text} onChange={(e) => setText(e.target.value)} />
      <button onClick={handleSubmit}>Analyze Sentiment</button>
      <p>Sentiment: {sentiment}</p>
    </div>
  );
}

export default App;
