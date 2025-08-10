// src/App.jsx
import React, { useEffect, useRef, useState } from "react";
import { Holistic } from "@mediapipe/holistic";
import { Camera } from "@mediapipe/camera_utils";

/*
Features:
- Live camera visible
- MediaPipe Holistic for face/pose/hands
- Finger count (single hand)
- Eye open/closed detection (left/right EAR)
- Speaking detection (mouth opening + variance over time)
- Movement detection (nose delta)
- Canvas overlay with animated green lines/dots when actions happen
- Dynamic theme toggle
*/

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const cameraRef = useRef(null);
  const holisticRef = useRef(null);

  const [theme, setTheme] = useState("dark");
  const [fingerCount, setFingerCount] = useState(0);
  const [leftEyeOpen, setLeftEyeOpen] = useState(true);
  const [rightEyeOpen, setRightEyeOpen] = useState(true);
  const [speaking, setSpeaking] = useState(false);
  const [moving, setMoving] = useState(false);

  // buffers for mouth openness history for speaking detection
  const mouthBufRef = useRef([]);
  const lastNoseRef = useRef(null);
  const lastTimeRef = useRef(Date.now());
  const pulseRef = useRef(0);

  useEffect(() => {
    // Create Holistic instance and Camera
    const holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: true,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });

    holistic.onResults(onResults);
    holisticRef.current = holistic;

    cameraRef.current = new Camera(videoRef.current, {
      onFrame: async () => {
        try {
          await holistic.send({ image: videoRef.current });
        } catch (e) {
          // some browsers may throw if camera not ready yet
          // console.warn(e);
        }
      },
      width: 640,
      height: 480,
    });

    cameraRef.current.start();

    // animation pulse
    let mounted = true;
    function animate() {
      pulseRef.current += 0.04;
      if (mounted) requestAnimationFrame(animate);
    }
    animate();

    return () => {
      mounted = false;
      try { cameraRef.current && cameraRef.current.stop(); } catch {}
      try { holisticRef.current && holisticRef.current.close && holisticRef.current.close(); } catch {}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Helpers: chosen face-mesh indices (MediaPipe)
  // Right eye (inner coords) and left eye selections (common indices):
  // Right eye: [33, 160, 158, 133, 153, 144]
  // Left eye:  [362, 385, 387, 263, 373, 380]
  const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
  const LEFT_EYE  = [362, 385, 387, 263, 373, 380];

  // For mouth center we use landmarks 13 (upper inner lip) and 14 (lower inner lip)
  const MOUTH_UP = 13;
  const MOUTH_DOWN = 14;

  // Face width reference between left cheek (234) and right cheek (454) or 127 and 356 etc.
  const FACE_LEFT = 234;
  const FACE_RIGHT = 454;

  function onResults(results) {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = 640;
    canvas.height = 480;

    // draw flipped video (mirror) so it feels natural
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    if (results.image) ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    // extract landmarks
    const face = results.faceLandmarks || null; // array of 468 points
    const leftHand = results.leftHandLandmarks && results.leftHandLandmarks[0] ? results.leftHandLandmarks[0] : results.leftHandLandmarks ? results.leftHandLandmarks[0] : null;
    const rightHand = results.rightHandLandmarks && results.rightHandLandmarks[0] ? results.rightHandLandmarks[0] : results.rightHandLandmarks ? results.rightHandLandmarks[0] : null;
    // MediaPipe Holistic returns hands as arrays per hand; easier to check results.leftHandLandmarks/results.rightHandLandmarks
    const lhand = results.leftHandLandmarks;
    const rhand = results.rightHandLandmarks;

    // finger counting: prefer the detected hand (right hand preferred if both)
    let handLandmarks = null;
    if (rhand && rhand.length) handLandmarks = rhand[0];
    else if (lhand && lhand.length) handLandmarks = lhand[0];

    // count fingers if handLandmarks exists
    if (handLandmarks && handLandmarks.length === 21) {
      setFingerCount(countFingersFromHand(handLandmarks));
    } else {
      setFingerCount(0);
    }

    // Eyes open/closed via EAR
    if (face && face.length >= 468) {
      const rightEAR = computeEAR(face, RIGHT_EYE);
      const leftEAR = computeEAR(face, LEFT_EYE);
      setRightEyeOpen(rightEAR > 0.20); // thresholds tuned empirically
      setLeftEyeOpen(leftEAR > 0.20);

      // mouth openness (normalized)
      const mouthOpen = distance(face[MOUTH_UP], face[MOUTH_DOWN]) / Math.max(1e-6, distance(face[FACE_LEFT], face[FACE_RIGHT]));
      // push to buffer
      const mBuf = mouthBufRef.current;
      const now = Date.now();
      if (mBuf.length > 30) mBuf.shift();
      mBuf.push({ t: now, v: mouthOpen });
      mouthBufRef.current = mBuf;

      // speaking detection: mouthOpen average + recent variance/jerk
      const avg = mBuf.reduce((s, x) => s + x.v, 0) / mBuf.length;
      const diffs = mBuf.slice(1).map((x,i) => Math.abs(x.v - mBuf[i].v));
      const meanDiff = diffs.length ? diffs.reduce((s,d)=>s+d,0)/diffs.length : 0;
      setSpeaking(avg > 0.03 && meanDiff > 0.008); // thresholds tuned for webcam; adjust if necessary
    } else {
      setRightEyeOpen(true);
      setLeftEyeOpen(true);
      setSpeaking(false);
    }

    // movement detection: use pose nose if present else face center
    const pose = results.poseLandmarks || null;
    let nose = null;
    if (pose && pose.length > 0) {
      nose = pose[0]; // pose landmark 0 is nose
    } else if (face && face.length > 1) {
      nose = face[1]; // approximate nose
    }

    if (nose) {
      const prev = lastNoseRef.current;
      if (prev) {
        const dx = Math.abs(nose.x - prev.x);
        const dy = Math.abs(nose.y - prev.y);
        const speed = Math.sqrt(dx*dx + dy*dy);
        setMoving(speed > 0.003); // tuned threshold
      }
      lastNoseRef.current = { x: nose.x, y: nose.y };
    } else {
      setMoving(false);
    }

    // Draw overlay: landmarks and animated green lines when events active
    drawOverlay(ctx, { face, handLandmarks: handLandmarks || null, rhand, lhand });
  }

  // --- Utilities ---

  // compute Euclidean distance in normalized coords
  function distance(a,b) {
    const dx = (a.x - b.x);
    const dy = (a.y - b.y);
    return Math.sqrt(dx*dx + dy*dy);
  }

  // Eye aspect ratio using 6 points: [p1,p2,p3,p4,p5,p6]
  function computeEAR(face, indices) {
    try {
      const p1 = face[indices[0]], p2 = face[indices[1]], p3 = face[indices[2]],
            p4 = face[indices[3]], p5 = face[indices[4]], p6 = face[indices[5]];
      // use vertical distances and horizontal
      const A = distance(p2, p6);
      const B = distance(p3, p5);
      const C = distance(p1, p4);
      const ear = (A + B) / (2.0 * Math.max(1e-6, C));
      return ear;
    } catch (e) {
      return 0.3; // default open
    }
  }

  // Count fingers from a single hand landmarks array (21 points)
  function countFingersFromHand(landmarks) {
    // TIP indices: thumb:4, index:8, middle:12, ring:16, pinky:20
    const TIP = [4,8,12,16,20];
    const PIP = [2,6,10,14,18]; // for thumb use different check
    // for index..pinky: tip.y < pip.y => finger up (normalized coords: y downwards)
    let count = 0;
    try {
      // Index to pinky
      for (let i=1;i<=4;i++) {
        if (landmarks[TIP[i]].y < landmarks[PIP[i]].y) count++;
      }
      // Thumb: check x vs ip and hand orientation: if thumb tip.x relatively left/right
      // Simple heuristic: thumb extended if tip.x is away from palm center (landmark 0)
      const thumbTip = landmarks[4];
      const thumbIp = landmarks[3];
      const palmX = landmarks[0].x;
      // if tip.x is less than palmX by a margin -> left direction on normalized coords (camera mirrored)
      if (Math.abs(thumbTip.x - palmX) > 0.03) {
        // relative direction
        if ((thumbTip.x < palmX && thumbIp.x < palmX) || (thumbTip.x > palmX && thumbIp.x > palmX)) {
          // extended
          count++;
        }
      }
      return count;
    } catch (e) {
      return 0;
    }
  }

  // Draw overlay with animations
  function drawOverlay(ctx, { face, handLandmarks, rhand, lhand }) {
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;

    // mirror context for drawing consistency (we drew mirrored video already)
    ctx.save();
    ctx.clearRect(0,0,w,h);

    // draw mirrored video underlay (video already drawn in results loop, but ensure)
    // We rely on results.image being drawn earlier - to keep shapes in sync we'll not re-draw image here.

    // draw face landmarks
    if (face) {
      // draw a soft mesh glow when speaking / eyes closed / moving
      const isSpeaking = speaking;
      const isMoving = moving;
      const eyeClosed = (!leftEyeOpen) || (!rightEyeOpen);

      // draw green animated lines connecting some facial landmarks if speaking
      ctx.lineWidth = 2;
      // dynamic pulse
      const pulse = 1 + Math.sin(pulseRef.current) * 0.35;

      // draw small dots for chosen landmarks (eyes/mouth)
      const drawPt = (pt,color,r) => {
        const x = (1 - pt.x) * w;
        const y = pt.y * h;
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.95;
        ctx.arc(x,y,r * pulse,0,Math.PI*2);
        ctx.fill();
      };

      // eyes
      RIGHT_EYE.forEach(idx => drawPt(face[idx], rightEyeOpen ? "rgba(0,220,120,0.9)" : "rgba(220,60,60,0.95)", 3));
      LEFT_EYE.forEach(idx => drawPt(face[idx], leftEyeOpen ? "rgba(0,220,120,0.9)" : "rgba(220,60,60,0.95)", 3));

      // mouth center
      drawPt(face[MOUTH_UP], speaking ? "rgba(0,255,140,0.95)" : "rgba(255,200,0,0.9)", 4);
      drawPt(face[MOUTH_DOWN], speaking ? "rgba(0,255,140,0.95)" : "rgba(255,200,0,0.9)", 4);

      // If speaking, draw connecting green lines around mouth
      if (speaking) {
        ctx.strokeStyle = "rgba(0,255,150,0.9)";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        const mouthIndices = [61,146,91,181,84,17,314,405,321,375,291,308]; // an outer mouth ring (common facemesh indices)
        mouthIndices.forEach((mi, i) => {
          const p = face[mi];
          const x = (1 - p.x) * w;
          const y = p.y * h;
          if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        });
        ctx.closePath();
        ctx.stroke();

        // glow
        ctx.strokeStyle = "rgba(0,255,150,0.15)";
        ctx.lineWidth = 18 * pulse;
        ctx.stroke();
      }

      // if eyes closed, draw red rings around eyes
      if (!leftEyeOpen || !rightEyeOpen) {
        ctx.strokeStyle = "rgba(255,60,60,0.9)";
        ctx.lineWidth = 3;
        // right eye region bounding box
        const rePts = RIGHT_EYE.map(i => ({ x:(1-face[i].x)*w, y:face[i].y*h }));
        drawPolygon(ctx, rePts);
        const lePts = LEFT_EYE.map(i => ({ x:(1-face[i].x)*w, y:face[i].y*h }));
        drawPolygon(ctx, lePts);
      }
    }

    // draw hand landmarks + skeleton
    if (handLandmarks && handLandmarks.length === 21) {
      // draw skeleton lines
      ctx.strokeStyle = "rgba(0,200,120,0.95)";
      ctx.lineWidth = 2;
      const toPt = (lm) => ({ x: (1 - lm.x) * w, y: lm.y * h });
      const connections = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [0,9],[9,10],[10,11],[11,12],
        [0,13],[13,14],[14,15],[15,16],
        [0,17],[17,18],[18,19],[19,20]
      ];
      ctx.beginPath();
      connections.forEach(([a,b]) => {
        const A = toPt(handLandmarks[a]);
        const B = toPt(handLandmarks[b]);
        ctx.moveTo(A.x,A.y);
        ctx.lineTo(B.x,B.y);
      });
      ctx.stroke();

      // draw fingertips highlighted when finger up
      const tipped = [4,8,12,16,20];
      tipped.forEach((i, idx) => {
        const lm = handLandmarks[i];
        const p = { x: (1 - lm.x) * w, y: lm.y * h };
        // simple check if finger is up vs pip
        // draw green if up else gray
        const pip = handLandmarks[[2,6,10,14,18][idx]];
        const isUp = pip && lm && (lm.y < pip.y);
        ctx.beginPath();
        ctx.fillStyle = isUp ? "rgba(0,255,140,0.95)" : "rgba(200,200,200,0.85)";
        ctx.globalAlpha = 0.95;
        ctx.arc(p.x,p.y, isUp ? 7 * (1 + Math.sin(pulseRef.current)/6) : 5, 0, Math.PI*2);
        ctx.fill();
      });
    }

    // small HUD: finger count, eye, speak, moving
    drawHUD(ctx);

    ctx.restore();
  }

  function drawHUD(ctx) {
    const w = ctx.canvas.width;
    // panel background
    const pad = 12;
    const width = 280, height = 110;
    const x = w - width - pad, y = pad;

    ctx.save();
    ctx.globalAlpha = theme === "dark" ? 0.75 : 0.95;
    ctx.fillStyle = theme === "dark" ? "#000000" : "#ffffff";
    roundRect(ctx, x, y, width, height, 10, true, false);
    ctx.globalAlpha = 1;

    ctx.fillStyle = theme === "dark" ? "#fff" : "#111";
    ctx.font = "14px Inter, Arial";
    ctx.fillText(`Fingers: ${fingerCount}`, x + 14, y + 28);

    ctx.fillStyle = leftEyeOpen ? "#0fbf5f" : "#ef5350";
    ctx.fillText(`Left eye: ${leftEyeOpen ? "Open" : "Closed"}`, x + 14, y + 50);

    ctx.fillStyle = rightEyeOpen ? "#0fbf5f" : "#ef5350";
    ctx.fillText(`Right eye: ${rightEyeOpen ? "Open" : "Closed"}`, x + 14, y + 72);

    ctx.fillStyle = speaking ? "#0fbf5f" : "#888";
    ctx.fillText(`Speaking: ${speaking ? "Yes" : "No"}`, x + 140, y + 28);

    ctx.fillStyle = moving ? "#0fbf5f" : "#888";
    ctx.fillText(`Moving: ${moving ? "Yes" : "No"}`, x + 140, y + 50);

    ctx.restore();
  }

  // small drawing helpers
  function drawPolygon(ctx, pts) {
    if (!pts || !pts.length) return;
    ctx.beginPath();
    pts.forEach((p,i) => i===0?ctx.moveTo(p.x,p.y):ctx.lineTo(p.x,p.y));
    ctx.closePath();
    ctx.stroke();
  }

  function roundRect(ctx, x, y, w, h, r = 8, fill = true, stroke = true) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    if (fill) ctx.fill();
    if (stroke) ctx.stroke();
  }

  // UI render
  return (
    <div style={theme === "dark" ? styles.pageDark : styles.pageLight}>
      <div style={styles.header}>
        <div style={styles.brand}>Holistic Vision — Live</div>
        <div style={styles.controls}>
          <button style={styles.themeBtn} onClick={() => setTheme(t => t === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "Light" : "Dark"} Theme
          </button>
        </div>
      </div>

      <main style={styles.main}>
        <div style={styles.leftCol}>
          <div style={styles.card}>
            <div style={{fontSize:18,fontWeight:700}}>Live Camera</div>
            <div style={{marginTop:10, color:"#888"}}>Allow camera permissions if asked.</div>

            <div style={styles.viewer}>
              <video ref={videoRef} style={styles.video} playsInline autoPlay muted />
              <canvas ref={canvasRef} style={styles.canvas} />
            </div>

            <div style={styles.tips}>
              <div>• Show your hand to count fingers.</div>
              <div>• Open/close eyes to see detection.</div>
              <div>• Speak to see mouth detection & animation.</div>
            </div>
          </div>
        </div>

        <div style={styles.rightCol}>
          <div style={styles.panel}>
            <h3>Status</h3>
            <div style={styles.statusRow}>
              <StatusChip label={`Fingers: ${fingerCount}`} active />
              <StatusChip label={`Left Eye`} active={leftEyeOpen} />
              <StatusChip label={`Right Eye`} active={rightEyeOpen} />
              <StatusChip label={`Speaking`} active={speaking} />
              <StatusChip label={`Moving`} active={moving} />
            </div>

            <div style={{marginTop:12}}>
              <strong>Notes</strong>
              <ul style={{marginTop:8,color:"#666"}}>
                <li>Works best with good lighting and the face/palm facing the camera.</li>
                <li>Thresholds are heuristic — you can tweak EAR & mouth thresholds if needed.</li>
                <li>For two-hand counts or fine mouth/speech detection you can expand the logic.</li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

// small status chip
function StatusChip({ label, active = false }) {
  return (
    <div style={{
      padding: "8px 12px",
      borderRadius: 999,
      background: active ? "linear-gradient(90deg,#0fbf5f,#00d27a)" : "#eee",
      color: active ? "#062b1f" : "#666",
      fontWeight: 600,
      marginRight: 8,
      minWidth: 92,
      textAlign: "center",
    }}>
      {label}
    </div>
  );
}

const styles = {
  pageDark: {
    background: "#0b1220",
    minHeight: "100vh",
    color: "#e6eef6",
    fontFamily: "Inter, system-ui, Arial, sans-serif",
    padding: 20
  },
  pageLight: {
    background: "#f3f6fb",
    minHeight: "100vh",
    color: "#111827",
    fontFamily: "Inter, system-ui, Arial, sans-serif",
    padding: 20
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 18
  },
  brand: { fontWeight: 800, fontSize: 20 },
  controls: {},
  themeBtn: {
    padding: "8px 12px",
    borderRadius: 8,
    border: "none",
    cursor: "pointer",
    background: "linear-gradient(90deg,#3b82f6,#06b6d4)",
    color: "#fff",
    fontWeight: 700
  },
  main: { display: "flex", gap: 20 },
  leftCol: { flex: 1 },
  rightCol: { width: 360 },
  card: {
    background: "linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01))",
    padding: 16,
    borderRadius: 12,
    boxShadow: "0 10px 30px rgba(2,6,23,0.5)"
  },
  viewer: {
    marginTop: 16,
    position: "relative",
    width: 640,
    maxWidth: "100%",
    borderRadius: 12,
    overflow: "hidden",
    background: "#000",
    display: "inline-block"
  },
  video: {
    width: "100%",
    height: "auto",
    display: "block",
    position: "absolute",
    left: 0,
    top: 0,
    transform: "scaleX(-1)", // mirror so user sees mirror
    WebkitTransform: "scaleX(-1)"
  },
  canvas: {
    position: "relative",
    width: "100%",
    height: "auto",
    pointerEvents: "none", // allow clicks through
    display: "block"
  },
  tips: { marginTop: 12, color: "#9fb0c8" },
  panel: {
    background: "linear-gradient(180deg,#021124, rgba(255,255,255,0.02))",
    padding: 16,
    borderRadius: 12,
    boxShadow: "0 10px 30px rgba(2,6,23,0.6)"
  },
  statusRow: { display: "flex", flexWrap: "wrap", gap: 10, marginTop: 8 }
};
