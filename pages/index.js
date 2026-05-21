import { useState } from "react";

const PAINTING_ID = 1;   // hardcoded for now; later comes from route/props

export default function Home() {
  const [file,          setFile]          = useState(null);
  const [aiResults,     setAiResults]     = useState(null);
  const [crowdStats,    setCrowdStats]    = useState(null);
  const [guessEmotion,  setGuessEmotion]  = useState("");
  const [submitted,     setSubmitted]     = useState(false);
  const [loading,       setLoading]       = useState(false);

  // 1. User submits their emotion guess BEFORE seeing AI result
  async function handleGuess() {
    if (!guessEmotion) return alert("Pick an emotion first.");
    await fetch("/api/annotate", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ painting_id: PAINTING_ID, guessed_emotion: guessEmotion }),
    });
    setSubmitted(true);

    // 2. Now fetch AI prediction + crowd stats in parallel
    setLoading(true);
    const testUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg";

    const [predictRes, statsRes] = await Promise.all([
      fetch("/api/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ image_url: testUrl }),
      }),
      fetch(`/api/annotate?painting_id=${PAINTING_ID}`),
    ]);

    setAiResults((await predictRes.json()).predictions);
    setCrowdStats(await statsRes.json());
    setLoading(false);
  }

  return (
    <div>
      <h1>Computational Aesthetics</h1>

      <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files[0])} />

      {/* Step 1 – human guess */}
      {!submitted && (
        <div>
          <p>What emotion does this painting express?</p>
          <select value={guessEmotion} onChange={(e) => setGuessEmotion(e.target.value)}>
            <option value="">-- pick one --</option>
            {["joy","sadness","fear","anger","awe","calmness","disgust","surprise"].map(e => (
              <option key={e} value={e}>{e}</option>
            ))}
          </select>
          <button onClick={handleGuess}>Submit guess & Analyze</button>
        </div>
      )}

      {loading && <p>Loading AI prediction…</p>}

      {/* Step 2 – reveal AI result + crowd stats */}
      {submitted && !loading && (
        <div>
          <h2>AI Prediction</h2>
          {aiResults?.map(r => (
            <p key={r.emotion}>{r.emotion}: {r.confidence}</p>
          ))}

          <h2>Crowd Stats</h2>
          {crowdStats && (
            <div>
              <p>Total votes: {crowdStats.total_votes}</p>
              <p>Top label:   {crowdStats.top_label}</p>
              {Object.entries(crowdStats.label_counts).map(([emotion, count]) => (
                <p key={emotion}>{emotion}: {count}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}