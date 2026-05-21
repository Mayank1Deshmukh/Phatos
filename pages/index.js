import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleAnalyze() {
    if (!file) return alert("Please select an image first.");

    // For now, use a test URL. Later you can upload the file to storage
    // and use the returned URL instead.
    const testUrl = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg";

    setLoading(true);
    setResults(null);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_url: testUrl }),
      });
      const data = await res.json();
      setResults(data.predictions);
    } catch (err) {
      setResults([{ emotion: "Error", confidence: err.message }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h1>Computational Aesthetics</h1>
      <input
        type="file"
        accept="image/*"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={handleAnalyze} disabled={loading}>
        {loading ? "Analyzing…" : "Analyze Emotion"}
      </button>

      <div>
        {results &&
          results.map((r) => (
            <p key={r.emotion}>
              {r.emotion}: {r.confidence}
            </p>
          ))}
      </div>
    </div>
  );
}