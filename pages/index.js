import { useState } from "react";

const PAINTING_ID = 1;

export default function Home() {
  const [file, setFile] = useState(null);
  const [aiResults, setAiResults] = useState(null);
  const [crowdStats, setCrowdStats] = useState(null);
  const [guessEmotion, setGuessEmotion] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);

  async function handleGuess() {
    if (!guessEmotion) return alert("Pick an emotion first.");

    const testUrl =
      "https://images.unsplash.com/photo-1777895868494-4e01af8487b6?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxmZWF0dXJlZC1waG90b3MtZmVlZHw3fHx8ZW58MHx8fHx8";

    setLoading(true);

    const annotatePromise = fetch("/api/annotate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        painting_id: PAINTING_ID,
        guessed_emotion: guessEmotion,
      }),
    });

    const predictPromise = fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_url: testUrl }),
    });

    const statsPromise = fetch(`/api/annotate?painting_id=${PAINTING_ID}`);

    await annotatePromise;

    const [predictRes, statsRes] = await Promise.all([predictPromise, statsPromise]);
    const predictJson = await predictRes.json();
    const statsJson = await statsRes.json();

    setAiResults(predictJson.predictions);
    setCrowdStats(statsJson);
    setSubmitted(true);
    setLoading(false);
  }

  return (
    <div style={{ padding: 24, maxWidth: 720, margin: "0 auto", fontFamily: "sans-serif" }}>
      <h1>Computational Aesthetics</h1>

      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />

      {!submitted && (
        <div style={{ marginTop: 20 }}>
          <p>What emotion does this painting express?</p>
          <select value={guessEmotion} onChange={(e) => setGuessEmotion(e.target.value)}>
            <option value="">-- pick one --</option>
            {["awe", "amusement", "contentment", "excitement", "anger", "fear", "sadness"].map(
              (e) => (
                <option key={e} value={e}>
                  {e}
                </option>
              )
            )}
          </select>
          <button style={{ marginLeft: 12 }} onClick={handleGuess}>
            Submit guess & Analyze
          </button>
        </div>
      )}

      {loading && <p style={{ marginTop: 20 }}>Loading AI prediction…</p>}

      {submitted && !loading && (
        <div style={{ marginTop: 24 }}>
          <h2>AI Prediction</h2>
          {aiResults?.map((r) => (
            <p key={r.emotion}>
              {r.emotion}: {r.confidence}
            </p>
          ))}

          <h2 style={{ marginTop: 24 }}>Crowd Stats</h2>
          {crowdStats && (
            <div>
              <p>Total votes: {crowdStats.total_votes}</p>
              <p>Top label: {crowdStats.top_label}</p>
              {Object.entries(crowdStats.label_counts || {}).map(([emotion, count]) => (
                <p key={emotion}>
                  {emotion}: {count}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}