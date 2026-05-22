import { useState, useRef } from "react";

const EMOTIONS = ["awe", "amusement", "contentment", "excitement", "anger", "fear", "sadness"];

export default function Home() {
  const [imageSrc,     setImageSrc]     = useState(null);   // data-URL for preview
  const [imageB64,     setImageB64]     = useState(null);   // raw base64 for API
  const [guessEmotion, setGuessEmotion] = useState("");
  const [step,         setStep]         = useState("upload"); // upload | guess | result
  const [loading,      setLoading]      = useState(false);
  const [result,       setResult]       = useState(null);    // { emotion, confidence, heatmap_overlay, reasoning }
  const fileRef = useRef();

  // ── file pick ──────────────────────────────────────────────────────────────
  function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const dataUrl = ev.target.result;             // "data:image/...;base64,..."
      setImageSrc(dataUrl);
      setImageB64(dataUrl.split(",")[1]);            // strip prefix, keep raw base64
      setStep("guess");
      setResult(null);
    };
    reader.readAsDataURL(file);
  }

  // ── submit guess + fetch AI ────────────────────────────────────────────────
  async function handleSubmit() {
    if (!guessEmotion) return alert("Pick an emotion first.");
    setLoading(true);

    // fire annotate (don't await — best-effort)
    fetch("/api/annotate", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ painting_id: 0, guessed_emotion: guessEmotion }),
    });

    try {
      const res  = await fetch("/api/explain", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ image_b64: imageB64 }),
      });
      const data = await res.json();
      if (!res.ok) {
        alert(`Server error ${res.status}: ${data.detail ?? JSON.stringify(data)}`);
        setLoading(false);
        return;
      }
      setResult(data);
      setStep("result");
    } catch (err) {
      alert(`Something went wrong — ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setImageSrc(null); setImageB64(null);
    setGuessEmotion(""); setStep("upload"); setResult(null);
    if (fileRef.current) fileRef.current.value = "";
  }

  // ── styles (inline, no extra deps) ────────────────────────────────────────
  const s = {
    page:      { padding: 32, maxWidth: 900, margin: "0 auto", fontFamily: "sans-serif" },
    row:       { display: "flex", gap: 24, marginTop: 24 },
    col:       { flex: 1 },
    img:       { width: "100%", borderRadius: 8, display: "block" },
    label:     { fontSize: 13, color: "#555", marginBottom: 6 },
    emotion:   { fontSize: 22, fontWeight: 700, marginBottom: 4 },
    bar_wrap:  { background: "#eee", borderRadius: 4, height: 10, marginBottom: 16 },
    bar:       (pct) => ({ width: `${pct}%`, background: "#4f46e5", height: 10, borderRadius: 4 }),
    reasoning: { lineHeight: 1.7, color: "#222", marginTop: 8 },
    btn:       { padding: "10px 22px", background: "#4f46e5", color: "#fff",
                 border: "none", borderRadius: 6, cursor: "pointer", fontSize: 15 },
    ghost:     { padding: "10px 22px", background: "none", color: "#4f46e5",
                 border: "1px solid #4f46e5", borderRadius: 6, cursor: "pointer", fontSize: 15 },
    select:    { padding: "8px 12px", fontSize: 15, borderRadius: 6,
                 border: "1px solid #ccc", marginRight: 12 },
  };

  return (
    <div style={s.page}>
      <h1 style={{ marginBottom: 4 }}>Computational Aesthetics</h1>
      <p style={{ color: "#555", marginTop: 0 }}>
        Upload a painting — guess its emotion — then see what the model attends to.
      </p>

      {/* ── step 1: upload ── */}
      {step === "upload" && (
        <div style={{ marginTop: 24 }}>
          <input ref={fileRef} type="file" accept="image/*" onChange={handleFile} />
        </div>
      )}

      {/* ── step 2: guess ── */}
      {step === "guess" && (
        <div>
          <img src={imageSrc} alt="uploaded painting" style={{ ...s.img, maxWidth: 420, marginTop: 20 }} />
          <div style={{ marginTop: 20 }}>
            <p style={{ marginBottom: 10 }}>What emotion does this painting express?</p>
            <select style={s.select} value={guessEmotion} onChange={e => setGuessEmotion(e.target.value)}>
              <option value="">-- pick one --</option>
              {EMOTIONS.map(e => <option key={e} value={e}>{e}</option>)}
            </select>
            <button style={s.btn} onClick={handleSubmit} disabled={loading}>
              {loading ? "Analyzing…" : "Submit & Analyze"}
            </button>
          </div>
        </div>
      )}

      {/* ── step 3: result ── */}
      {step === "result" && result && (
        <div>
          {/* side-by-side images */}
          <div style={s.row}>
            <div style={s.col}>
              <p style={s.label}>Original</p>
              <img src={imageSrc} alt="original" style={s.img} />
            </div>
            <div style={s.col}>
              <p style={s.label}>Saliency heatmap</p>
              {result.heatmap_overlay
                ? <img
                    src={`data:image/png;base64,${result.heatmap_overlay}`}
                    alt="heatmap"
                    style={s.img}
                  />
                : <p style={{ color: "#999" }}>Heatmap unavailable</p>
              }
            </div>
          </div>

          {/* emotion + confidence */}
          <div style={{ marginTop: 28 }}>
            <p style={s.label}>Predicted emotion</p>
            <p style={s.emotion}>
              {result.emotion}&nbsp;
              <span style={{ fontSize: 16, fontWeight: 400, color: "#555" }}>
                {result.confidence != null
                  ? `(${(result.confidence * 100).toFixed(1)}% confidence)`
                  : ""}
              </span>
            </p>
            <div style={s.bar_wrap}>
              <div style={s.bar((result.confidence ?? 0) * 100)} />
            </div>
            <p style={{ color: "#555", fontSize: 14 }}>
              Your guess: <strong>{guessEmotion}</strong>
              {guessEmotion === result.emotion
                ? " ✓ matches the model"
                : " — the model disagreed"}
            </p>
          </div>

          {/* Gemini reasoning */}
          <div style={{ marginTop: 24, borderTop: "1px solid #eee", paddingTop: 20 }}>
            <p style={s.label}>Art-historical interpretation</p>
            <p style={s.reasoning}>{result.reasoning}</p>
          </div>

          <button style={{ ...s.ghost, marginTop: 28 }} onClick={reset}>
            Try another painting
          </button>
        </div>
      )}
    </div>
  );
}