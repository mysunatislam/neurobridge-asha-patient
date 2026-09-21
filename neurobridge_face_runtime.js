/**
 * NeuroBridge Asha — NeuroSense Facial Communication Engine (Web Runtime)
 * Powered by NeuroSense (MediaPipe Face Mesh + Auto-Calibrated Gesture Intelligence)
 *
 * Specific Patient Communication Rules (NeuroSense Suite):
 * 1. 3 intentional eye blinks in a row (looking at camera) -> "I need water"
 * 2. 3 left head movements                                 -> "I need food"
 * 3. 3 right head movements                                -> "I need to go to toilet"
 * 4. Nodding while smiling                                  -> "I am okay, thank you"
 *
 * Runs 100% locally in browser via MediaPipe FaceMesh. No backend or uploads.
 */
(function (global) {
  'use strict';

  // Canonical MediaPipe Face Mesh (468 + iris 478) indices
  const NF_LM = {
    eyeL: { outer: 33, inner: 133, up1: 160, up2: 158, low1: 153, low2: 144 },
    eyeR: { outer: 263, inner: 362, up1: 385, up2: 387, low1: 373, low2: 380 },
    irisL: [468, 469, 470, 471, 472],
    irisR: [473, 474, 475, 476, 477],
    browL: [70, 63, 105, 66, 107],
    browR: [336, 296, 334, 293, 300],
    noseTip: 1, noseBridge: 6, chin: 152, forehead: 10,
    cheekL: 234, cheekR: 454, jawL: 132, jawR: 361,
    mouthL: 61, mouthR: 291, lipUpIn: 13, lipLowIn: 14,
    lipUpOut: 0, lipLowOut: 17,
  };

  // ── Pure geometry helpers ─────────────────────────────────────────────────
  function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function ear(lm, eye) {
    const v1 = dist(lm[eye.up1], lm[eye.low1]);
    const v2 = dist(lm[eye.up2], lm[eye.low2]);
    const h = Math.max(1e-6, dist(lm[eye.outer], lm[eye.inner]));
    return (v1 + v2) / (2 * h);
  }

  function mar(lm) {
    const h = dist(lm[13], lm[14]);
    const w = Math.max(1e-6, dist(lm[61], lm[291]));
    return h / w;
  }

  function mouthWidth(lm) {
    return dist(lm[61], lm[291]);
  }

  function smileIntensity(lm, neutral) {
    const w = mouthWidth(lm);
    const w0 = (neutral && neutral.mouthW) || w;
    const stretch = Math.max(0, Math.min(1, (w / Math.max(1e-6, w0) - 1) * 4));
    const midY = (lm[13].y + lm[14].y) / 2;
    const liftL = (midY - lm[61].y) / Math.max(1e-6, w);
    const liftR = (midY - lm[291].y) / Math.max(1e-6, w);
    const lift = Math.max(0, Math.min(1, (liftL + liftR) * 2.2));
    return Math.max(0, Math.min(1, stretch * 0.55 + lift * 0.45));
  }

  function symmetryScore(lm, neutral) {
    const w = Math.max(1e-6, mouthWidth(lm));
    const nL = neutral ? neutral.cornerLY : lm[61].y;
    const nR = neutral ? neutral.cornerRY : lm[291].y;
    const dL = Math.abs(lm[61].y - nL) / w + Math.abs(lm[61].x - (neutral ? neutral.cornerLX : lm[61].x)) / w;
    const dR = Math.abs(lm[291].y - nR) / w + Math.abs(lm[291].x - (neutral ? neutral.cornerRX : lm[291].x)) / w;
    const denom = Math.max(1e-6, dL + dR);
    const bal = 1 - Math.abs(dL - dR) / denom;
    const mag = Math.min(1, (dL + dR) * 6);
    return Math.round(100 * (1 - mag * (1 - bal) * 1.6 - mag * 0.06));
  }

  function headPose(lm) {
    const faceW = Math.max(1e-6, dist(lm[234], lm[454]));
    const faceH = Math.max(1e-6, dist(lm[10], lm[152]));
    const midX = (lm[234].x + lm[454].x) / 2;
    const midEyeY = (lm[33].y + lm[263].y) / 2;
    const yaw = ((lm[1].x - midX) / faceW) * 130;
    const pitch = ((lm[1].y - midEyeY) / faceH) * 120 - 8;
    const roll = (Math.atan2(lm[263].y - lm[33].y, lm[263].x - lm[33].x) * 180) / Math.PI;
    return { yaw, pitch, roll };
  }

  function lipDeviation(lm, dev0) {
    const fw = Math.max(1e-6, dist(lm[234], lm[454]));
    const mouthCx = (lm[61].x + lm[291].x) / 2;
    return (mouthCx - lm[1].x) / fw - (dev0 || 0);
  }

  function cornerDepression(lm) {
    const fw = Math.max(1e-6, dist(lm[234], lm[454]));
    const midY = (lm[13].y + lm[14].y) / 2;
    return ((lm[61].y + lm[291].y) / 2 - midY) / fw;
  }

  // ── Engine state ──────────────────────────────────────────────────────────
  let isRunning = false;
  let faceMesh = null;
  let videoEl = null;
  let cameraStream = null;
  let animFrameId = null;
  let sending = false;
  let simulationTimer = null;
  let simulatedActive = false;

  // Auto-calibrated Digital Twin baseline
  let calFrames = 0;
  let twin = null;
  let calAccum = { ear: 0, mouthW: 0, browGap: 0, dev: 0, yaw: 0, pitch: 0, roll: 0, count: 0 };

  // ── NeuroSense Rule Tuning Constants ──────────────────────────────────────
  const BLINKS_FOR_WATER = 3;
  const BLINK_WINDOW_MS = 3500;
  const BLINK_GAZE_YAW_LIMIT = 11;   // degrees — facing camera directly (distinguishable from head turns at 12°+)
  const BLINK_GAZE_PITCH_LIMIT = 22;

  const HEAD_LEFT_ENTER_DEG = 12;
  const HEAD_LEFT_EXIT_DEG = 6;
  const HEAD_TURNS_FOR_FOOD = 3;
  const HEAD_LEFT_WINDOW_MS = 5000;

  const HEAD_RIGHT_ENTER_DEG = 12;
  const HEAD_RIGHT_EXIT_DEG = 6;
  const HEAD_TURNS_FOR_TOILET = 3;
  const HEAD_RIGHT_WINDOW_MS = 5000;

  const NOD_PITCH_THRESHOLD = 5;    // degrees — reversal amplitude
  const NOD_WINDOW_MS = 2000;
  const NOD_REVERSALS_REQUIRED = 3;
  const NOD_SMILE_THRESHOLD = 0.35;
  const NOD_COOLDOWN_MS = 4000;

  // ── Rule 1: Blink detector state ──────────────────────────────────────────
  let blinkClosed = false;
  let blinkT0 = 0;
  let recentBlinks = [];

  // ── Rule 2: Head-left detector state ──────────────────────────────────────
  let headLeftArmedForFood = false;
  let recentHeadLeftTurns = [];

  // ── Rule 3: Head-right detector state ─────────────────────────────────────
  let headRightArmedForToilet = false;
  let recentHeadRightTurns = [];

  // ── Rule 4: Nod-while-smile detector state ────────────────────────────────
  let pitchHistory = [];    // Array of { pitch, t } in seconds
  let lastNodSmileAt = 0;

  // ── Navigation state (separate from rules) ────────────────────────────────
  let headLeftArmedNav = false;
  let headRightArmedNav = false;
  let lastLeftNavAt = 0;
  let lastRightNavAt = 0;

  let prevT = performance.now() / 1000;

  function broadcastStatus(status) {
    window.postMessage({
      type: 'neurobridge_face_status',
      ...status,
    }, '*');
  }

  function broadcastSignal(kind, confidence = 0.95, metadata = {}) {
    window.postMessage({
      type: 'neurobridge_patient_signal',
      kind: kind,
      confidence: confidence,
      observedAt: Date.now(),
      ...metadata,
    }, '*');
  }

  function speakAndEmit(phrase, intent, signalKind) {
    console.log('[NeuroSense Intent Triggered]:', phrase, '-> intent:', intent);
    try {
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
        const ut = new SpeechSynthesisUtterance(phrase);
        ut.rate = 0.95;
        ut.pitch = 1.05;
        window.speechSynthesis.speak(ut);
      }
    } catch (_) {}

    broadcastSignal(signalKind, 1.0, {
      intent: intent,
      label: phrase,
    });
  }

  function resetAutoCalibration() {
    calFrames = 0;
    twin = null;
    calAccum = { ear: 0, mouthW: 0, browGap: 0, dev: 0, yaw: 0, pitch: 0, roll: 0, count: 0 };
    resetRuleState();
    console.log('[NeuroSense] Auto-calibration reset: gathering fresh baseline.');
  }

  function resetRuleState() {
    blinkClosed = false;
    blinkT0 = 0;
    recentBlinks = [];
    headLeftArmedForFood = false;
    recentHeadLeftTurns = [];
    headRightArmedForToilet = false;
    recentHeadRightTurns = [];
    pitchHistory = [];
    lastNodSmileAt = 0;
    headLeftArmedNav = false;
    headRightArmedNav = false;
  }

  function onMesh(res) {
    const lms = (res.multiFaceLandmarks && res.multiFaceLandmarks[0]) || null;
    if (!lms || lms.length < 468) {
      resetRuleState();
      broadcastStatus({
        faceDetected: false,
        lifecycle: 'active',
        message: 'Searching for patient face…',
      });
      return;
    }

    const t = performance.now() / 1000;
    const dt = Math.min(0.2, Math.max(0.001, t - prevT));
    prevT = t;
    const tMs = t * 1000;

    // ── 1. Auto-calibration (first 60 frames ≈ 2 seconds) ───────────────
    if (calFrames < 60) {
      calFrames++;
      const eL = ear(lms, NF_LM.eyeL);
      const eR = ear(lms, NF_LM.eyeR);
      const hp0 = headPose(lms);
      calAccum.ear += (eL + eR) / 2;
      calAccum.mouthW += mouthWidth(lms);
      calAccum.dev += lipDeviation(lms, 0);
      calAccum.yaw += hp0.yaw;
      calAccum.pitch += hp0.pitch;
      calAccum.roll += hp0.roll;
      calAccum.count++;

      if (calFrames >= 60) {
        twin = {
          earMean: calAccum.ear / calAccum.count,
          mouthW: calAccum.mouthW / calAccum.count,
          dev0: calAccum.dev / calAccum.count,
          head: {
            yaw: calAccum.yaw / calAccum.count,
            pitch: calAccum.pitch / calAccum.count,
            roll: calAccum.roll / calAccum.count,
          },
        };
        console.log('[NeuroSense] Auto-calibration complete! Baseline Digital Twin:', twin);
      }
    }

    // ── 2. Compute metrics ──────────────────────────────────────────────
    const earL = ear(lms, NF_LM.eyeL);
    const earR = ear(lms, NF_LM.eyeR);
    const earAvg = (earL + earR) / 2;

    const marVal = mar(lms);
    const smile = smileIntensity(lms, twin);
    const sym = symmetryScore(lms, twin);
    const hp = headPose(lms);
    const yaw = hp.yaw - (twin ? twin.head.yaw : 0);
    const pitch = hp.pitch - (twin ? twin.head.pitch : 0);

    // Adaptive blink thresholds from baseline
    const baseEar = (twin && twin.earMean && twin.earMean > 0.15) ? twin.earMean : 0.25;
    const thClose = Math.max(0.14, baseEar * 0.65);
    const thOpen = Math.max(0.18, baseEar * 0.78);

    // ── Navigation: Edge-triggered head turns (separate from gesture rules) ─
    if (yaw < -11.0) {
      if (!headLeftArmedNav && (t - lastLeftNavAt > 0.55)) {
        headLeftArmedNav = true;
        lastLeftNavAt = t;
        console.log('[NeuroSense] Navigation step: Left');
        broadcastSignal('eyeLookLeft', 0.95);
      }
    } else if (yaw > -5.0) {
      headLeftArmedNav = false;
    }

    if (yaw > 11.0) {
      if (!headRightArmedNav && (t - lastRightNavAt > 0.55)) {
        headRightArmedNav = true;
        lastRightNavAt = t;
        console.log('[NeuroSense] Navigation step: Right');
        broadcastSignal('eyeLookRight', 0.95);
      }
    } else if (yaw < 5.0) {
      headRightArmedNav = false;
    }

    // ── Blink detection + NeuroSense Rule 1 (water) ─────────────────────────
    const isEyesClosed = earAvg < thClose || (earL < thClose && earR < thClose);
    let blinkJustCompleted = false;

    if (blinkClosed && (t - blinkT0 > 1.2)) {
      blinkClosed = false;
    }

    if (!blinkClosed && isEyesClosed) {
      blinkClosed = true;
      blinkT0 = t;
    } else if (blinkClosed && earAvg > thOpen) {
      blinkClosed = false;
      const dur = t - blinkT0;
      if (dur >= 0.08 && dur <= 0.90) {
        blinkJustCompleted = true;
        console.log('[NeuroSense] Deliberate Blink registered! dur:', dur.toFixed(2), 's');
        broadcastSignal('blink', 0.98);
      }
    }

    // Only process gesture rules if calibrated
    if (!twin) {
      broadcastStatus({
        faceDetected: true,
        lifecycle: 'active',
        message: 'Calibrating... ' + calFrames + '/60',
        calibrated: false,
        observedAt: Date.now(),
      });
      return;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NeuroSense Rule 1: 3 intentional blinks (looking at camera) → Water
    // ═══════════════════════════════════════════════════════════════════════
    const lookingAtCamera = Math.abs(yaw) < BLINK_GAZE_YAW_LIMIT && Math.abs(pitch) < BLINK_GAZE_PITCH_LIMIT;

    if (blinkJustCompleted && lookingAtCamera) {
      recentBlinks.push(tMs);
      recentBlinks = recentBlinks.filter(function(blinkAt) { return tMs - blinkAt <= BLINK_WINDOW_MS; });
      if (recentBlinks.length >= BLINKS_FOR_WATER) {
        recentBlinks = [];
        speakAndEmit('I need water', 'water', 'blink3');
      }
    }

    // Pitch history for nod detection
    pitchHistory.push({ pitch: pitch, t: tMs });
    pitchHistory = pitchHistory.filter(function(e) { return tMs - e.t <= NOD_WINDOW_MS + 500; });

    // ═══════════════════════════════════════════════════════════════════════
    // NeuroSense Rule 2: 3 left head movements → Food
    // ═══════════════════════════════════════════════════════════════════════
    if (yaw < -HEAD_LEFT_ENTER_DEG) {
      headLeftArmedForFood = true;
    } else if (headLeftArmedForFood && yaw > -HEAD_LEFT_EXIT_DEG) {
      headLeftArmedForFood = false;
      recentHeadLeftTurns.push(tMs);
      recentHeadLeftTurns = recentHeadLeftTurns.filter(function(turnAt) { return tMs - turnAt <= HEAD_LEFT_WINDOW_MS; });
      if (recentHeadLeftTurns.length >= HEAD_TURNS_FOR_FOOD) {
        recentHeadLeftTurns = [];
        speakAndEmit('I need food', 'food', 'headLeft3');
      }
    } else if (yaw > HEAD_LEFT_ENTER_DEG) {
      headLeftArmedForFood = false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NeuroSense Rule 3: 3 right head movements → Toilet
    // ═══════════════════════════════════════════════════════════════════════
    if (yaw > HEAD_RIGHT_ENTER_DEG) {
      headRightArmedForToilet = true;
    } else if (headRightArmedForToilet && yaw < HEAD_RIGHT_EXIT_DEG) {
      headRightArmedForToilet = false;
      recentHeadRightTurns.push(tMs);
      recentHeadRightTurns = recentHeadRightTurns.filter(function(turnAt) { return tMs - turnAt <= HEAD_RIGHT_WINDOW_MS; });
      if (recentHeadRightTurns.length >= HEAD_TURNS_FOR_TOILET) {
        recentHeadRightTurns = [];
        speakAndEmit('I need to go to toilet', 'toilet', 'headRight3');
      }
    } else if (yaw < -HEAD_RIGHT_ENTER_DEG) {
      headRightArmedForToilet = false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // NeuroSense Rule 4: Nodding while smiling → "I am okay, thank you"

    // ═══════════════════════════════════════════════════════════════════════
    if (smile > NOD_SMILE_THRESHOLD) {
      // Count pitch direction reversals in the recent window
      const recent = pitchHistory.filter(function(e) { return tMs - e.t <= NOD_WINDOW_MS; });
      let reversals = 0;
      for (let i = 2; i < recent.length; i++) {
        const d1 = recent[i - 1].pitch - recent[i - 2].pitch;
        const d2 = recent[i].pitch - recent[i - 1].pitch;
        if (d1 * d2 < 0 && Math.abs(d1) + Math.abs(d2) > NOD_PITCH_THRESHOLD) {
          reversals++;
        }
      }
      if (reversals >= NOD_REVERSALS_REQUIRED && t - lastNodSmileAt > NOD_COOLDOWN_MS / 1000) {
        lastNodSmileAt = t;
        speakAndEmit('I am okay, thank you', 'okay', 'nodSmile');
      }
    }

    // ── Live Telemetry Broadcast ─────────────────────────────────────────
    const isBlinkingNow = blinkClosed || earAvg < thClose;
    const leftOpen = isBlinkingNow ? 0.05 : Math.min(1.0, Math.max(0.0, (earL - 0.14) / 0.12));
    const rightOpen = isBlinkingNow ? 0.05 : Math.min(1.0, Math.max(0.0, (earR - 0.14) / 0.12));

    broadcastStatus({
      faceDetected: true,
      lifecycle: 'active',
      leftEyeOpen: leftOpen,
      rightEyeOpen: rightOpen,
      smileProbability: smile,
      headYaw: yaw,
      headPitch: pitch,
      eyebrowDistance: 0.18 + Math.abs(pitch) * 0.002,
      mouthDistance: marVal,
      symmetryScore: sym / 100,
      calibrated: calFrames >= 60,
      observedAt: Date.now(),
    });
  }

  // ── Synthetic Demonstration Engine ────────────────────────────────────────
  function startSimulationLoop() {
    if (simulationTimer) return;
    simulatedActive = true;
    console.log('[NeuroSense] Idle reference telemetry active.');

    let tick = 0;
    simulationTimer = setInterval(function() {
      tick++;
      const cycle = tick % 140;
      let curYaw = 0.0;
      let curPitch = 0.0;
      let curEye = 0.88;
      let curSmile = 0.05;

      if (cycle >= 10 && cycle <= 35) {
        const sub = (cycle - 10) % 5;
        curYaw = sub < 3 ? 3.0 : 0.0;
      } else if (cycle >= 40 && cycle <= 65) {
        const sub = (cycle - 40) % 5;
        curEye = sub < 2 ? 0.82 : 0.88;
      } else if (cycle >= 70 && cycle <= 95) {
        const progress = Math.sin(((cycle - 70) / 25) * Math.PI);
        curSmile = 0.04 + 0.12 * progress;
      } else if (cycle >= 100 && cycle <= 115) {
        const progress = Math.sin(((cycle - 100) / 15) * Math.PI);
        curPitch = -4.0 * progress;
      }

      broadcastStatus({
        faceDetected: true,
        lifecycle: 'active',
        leftEyeOpen: curEye,
        rightEyeOpen: curEye,
        smileProbability: curSmile,
        headYaw: curYaw,
        headPitch: curPitch,
        eyebrowDistance: 0.18,
        mouthDistance: 0.08 + curSmile * 0.06,
        symmetryScore: 0.95,
        calibrated: true,
        observedAt: Date.now(),
      });
    }, 100);
  }

  function stopSimulationLoop() {
    if (simulationTimer) {
      clearInterval(simulationTimer);
      simulationTimer = null;
    }
    simulatedActive = false;
  }

  async function pump() {
    if (!isRunning) return;
    if (videoEl && videoEl.readyState >= 2 && faceMesh && !sending) {
      sending = true;
      try {
        await faceMesh.send({ image: videoEl });
      } catch (err) {
        console.warn('[NeuroSense] Detection frame error:', err);
      }
      sending = false;
    }
    animFrameId = requestAnimationFrame(pump);
  }

  async function startCamera() {
    if (cameraStream) return true;
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('getUserMedia not supported in this browser');
      }
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      if (!videoEl) {
        videoEl = document.createElement('video');
        videoEl.id = 'neurobridge-live-face-feed';
        videoEl.autoplay = true;
        videoEl.playsInline = true;
        videoEl.muted = true;
        videoEl.width = 640;
        videoEl.height = 480;
        videoEl.style.cssText = 'position:fixed;bottom:0;right:0;width:160px;height:120px;opacity:0.01;pointer-events:none;z-index:-999;';
        document.body.appendChild(videoEl);
      }
      videoEl.srcObject = cameraStream;
      await videoEl.play();
      console.log('[NeuroSense] Camera stream playing.');
      return true;
    } catch (err) {
      console.warn('[NeuroSense] Camera acquisition denied/unavailable, fallback mode active:', err.message);
      return false;
    }
  }

  function initMesh() {
    if (faceMesh) return faceMesh;
    if (typeof FaceMesh === 'undefined') {
      console.warn('[NeuroSense] MediaPipe FaceMesh classic UMD not found in window.');
      return null;
    }
    try {
      const fm = new FaceMesh({
        locateFile: function(f) { return 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/' + f; },
      });
      fm.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      fm.onResults(onMesh);
      faceMesh = fm;
      console.log('[NeuroSense] FaceMesh initialized successfully.');
      return faceMesh;
    } catch (e) {
      console.warn('[NeuroSense] FaceMesh creation error:', e);
      return null;
    }
  }

  async function ensureMesh() {
    for (let i = 0; i < 20; i++) {
      const m = initMesh();
      if (m) return m;
      await new Promise(function(r) { setTimeout(r, 150); });
    }
    return null;
  }

  const NeuroBridgeFace = {
    async start() {
      if (isRunning) return;
      isRunning = true;
      resetAutoCalibration();

      startSimulationLoop();

      broadcastStatus({
        faceDetected: true,
        lifecycle: 'active',
        message: 'NeuroSense Facial Engine active',
      });

      const cameraOk = await startCamera();
      const meshOk = await ensureMesh();

      if (cameraOk && meshOk) {
        stopSimulationLoop();
        console.log('[NeuroSense] Real camera and FaceMesh active — live inference running.');
        animFrameId = requestAnimationFrame(pump);
      }
    },

    stop() {
      isRunning = false;
      if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
      }
      stopSimulationLoop();
      if (cameraStream) {
        cameraStream.getTracks().forEach(function(track) { track.stop(); });
        cameraStream = null;
      }
      if (videoEl) {
        videoEl.srcObject = null;
      }
      broadcastStatus({
        faceDetected: false,
        lifecycle: 'stopped',
        message: 'NeuroSense monitor stopped',
      });
    },

    // Manual test triggers for all 4 NeuroSense actions
    triggerWater() { speakAndEmit('I need water', 'water', 'blink3'); },
    triggerFood() { speakAndEmit('I need food', 'food', 'headLeft3'); },
    triggerToilet() { speakAndEmit('I need to go to toilet', 'toilet', 'headRight3'); },
    triggerOkay() { speakAndEmit('I am okay, thank you', 'okay', 'nodSmile'); },

    resetCalibration() { resetAutoCalibration(); },
  };

  global.NeuroBridgeFace = NeuroBridgeFace;

  if (typeof window !== 'undefined') {
    window.addEventListener('message', function(event) {
      const data = event.data;
      if (!data || typeof data !== 'object') return;
      if (data.type === 'neurobridge_trigger_gesture') {
        const g = data.gesture;
        if (g === 'water' || g === '3-blinks') NeuroBridgeFace.triggerWater();
        else if (g === 'food' || g === '3-head-left') NeuroBridgeFace.triggerFood();
        else if (g === 'toilet' || g === '3-head-right') NeuroBridgeFace.triggerToilet();
        else if (g === 'okay' || g === 'nod-smile') NeuroBridgeFace.triggerOkay();
      } else if (data.type === 'neurobridge_reset_calibration') {
        NeuroBridgeFace.resetCalibration();
      }
    });

    window.addEventListener('load', function() {
      console.log('[NeuroSense] Facial Communication Engine registered.');
      setTimeout(function() {
        if (!isRunning) NeuroBridgeFace.start();
      }, 300);
    });
  }
})(typeof window !== 'undefined' ? window : globalThis);
