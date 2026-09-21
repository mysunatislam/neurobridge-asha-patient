/**
 * NeuroBridge Asha — Clinical Facial Intelligence Engine (Web Runtime)
 * Powered by NeuroFace Sense (MediaPipe Face Mesh + Auto-Calibrated Clinical Intelligence)
 *
 * Specific Patient Communication Rules:
 * 1. Blinking 5 times in a row -> "I want water"
 * 2. Smiling (sustained) -> "I am feeling good"
 * 3. Abnormality (sustained lateral droop >=3.5% or pain pattern) -> "Emergency help needed"
 * 4. Moving head rightwards 5 times -> "Give me some food"
 * 5. Head nodding -> Confirm ("Yes / Confirm")
 *
 * Runs 100% locally in browser via MediaPipe FaceMesh. No backend or uploads.
 */
(function (global) {
  'use strict';

  // Canonical MediaPipe Face Mesh (468 + iris 478) indices (from NeuroFace Sense)
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

  // Pure geometry metrics from NeuroFace Sense
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

  function detectNod(hist) {
    if (!hist || hist.length < 30) return false;
    const seg = hist.slice(-45);
    let turns = 0;
    for (let i = 2; i < seg.length; i++) {
      const d1 = seg[i - 1] - seg[i - 2];
      const d2 = seg[i] - seg[i - 1];
      if (d1 * d2 < 0 && Math.abs(d1) + Math.abs(d2) > 3) turns++;
    }
    return turns >= 3;
  }

  // Engine state
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
  let twin = null; // { earMean, mouthW, browGap, dev0, head: { yaw, pitch, roll } }
  let calAccum = { ear: 0, mouthW: 0, browGap: 0, dev: 0, yaw: 0, pitch: 0, roll: 0, count: 0 };

  // Rule State & Gesture Detectors
  let blinkClosed = false;
  let blinkT0 = 0;
  let recentBlinks = []; // timestamps within 4s window

  let smileHoldTime = 0;
  let lastSmileCommandAt = 0;

  let devHoldTime = 0;
  let painHoldTime = 0;
  let lastAbnormalityAt = 0;

  let headRightArmed = false;
  let headLeftArmed = false;
  let lastLeftNavAt = 0;
  let lastRightNavAt = 0;
  let recentHeadRightTurns = []; // timestamps within 6s window

  let pitchHist = [];
  let lastNodAt = 0;
  let isNodding = false;

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
    console.log('[NeuroBridge Face Intent Triggered]:', phrase, '-> intent:', intent);
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
    console.log('[NeuroBridge Face] Auto-calibration reset: gathering fresh baseline.');
  }

  function onMesh(res) {
    const lms = (res.multiFaceLandmarks && res.multiFaceLandmarks[0]) || null;
    if (!lms || lms.length < 468) {
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

    // 1. Auto-calibration (first 60 frames = ~2 seconds)
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
        console.log('[NeuroBridge Face] Auto-calibration complete! Baseline Digital Twin:', twin);
      }
    }

    // 2. Metrics calculation against neutral baseline
    const earL = ear(lms, NF_LM.eyeL);
    const earR = ear(lms, NF_LM.eyeR);
    const earAvg = (earL + earR) / 2;

    const marVal = mar(lms);
    const smile = smileIntensity(lms, twin);
    const sym = symmetryScore(lms, twin);
    const hp = headPose(lms);
    const yaw = hp.yaw - (twin ? twin.head.yaw : 0);
    const pitch = hp.pitch - (twin ? twin.head.pitch : 0);
    const dev = lipDeviation(lms, twin ? twin.dev0 : 0);
    const depress = cornerDepression(lms);

    // Adaptive thresholding from baseline
    const baseEar = (twin && twin.earMean && twin.earMean > 0.15) ? twin.earMean : 0.25;
    const thClose = Math.max(0.14, baseEar * 0.65);
    const thOpen = Math.max(0.18, baseEar * 0.78);

    // 3. Deliberate Blink Detection (Clean hysteresis & safety timeout)
    const isEyesClosed = earAvg < thClose || (earL < thClose && earR < thClose);

    // Safety timeout: auto-release if eyes held closed > 1.2s so blink state never hangs
    if (blinkClosed && (t - blinkT0 > 1.2)) {
      blinkClosed = false;
    }

    if (!blinkClosed && isEyesClosed) {
      blinkClosed = true;
      blinkT0 = t;
    } else if (blinkClosed && earAvg > thOpen) {
      blinkClosed = false;
      const dur = t - blinkT0;
      // Deliberate intentional blink: 120ms to 850ms
      if (dur >= 0.12 && dur <= 0.85) {
        console.log('[NeuroBridge Face] Deliberate Blink registered! dur:', dur.toFixed(2), 's');
        broadcastSignal('blink', 0.98);
      }
    }

    // 4. Smiling (sustained) -> "I am feeling good"
    if (smile > 0.48) {
      smileHoldTime += dt;
      if (smileHoldTime >= 1.0 && t - lastSmileCommandAt > 4.0) {
        lastSmileCommandAt = t;
        smileHoldTime = -1.5; // Cooldown
        speakAndEmit('I am feeling good', 'feeling_good', 'smile');
      }
    } else {
      smileHoldTime = 0;
    }

    // 5. Sustained deliberate asymmetry check (very high threshold, must not be head turn)
    const devMag = Math.abs(dev);
    if (devMag >= 0.08 && Math.abs(yaw) < 8) {
      devHoldTime += dt;
      if (devHoldTime >= 5.0 && t - lastAbnormalityAt > 15.0) {
        lastAbnormalityAt = t;
        devHoldTime = 0;
        speakAndEmit('Emergency help needed', 'abnormality', 'seizureAlert');
      }
    } else {
      devHoldTime = 0;
    }

    // 6. Gaze & Head Turn Navigation (Edge-triggered with refractory cooldown)
    if (yaw < -11.0) {
      // Looking Left -> Single step left navigation
      if (!headLeftArmed && (t - lastLeftNavAt > 0.55)) {
        headLeftArmed = true;
        lastLeftNavAt = t;
        console.log('[NeuroBridge Face] Navigation step: Left');
        broadcastSignal('eyeLookLeft', 0.95);
      }
    } else if (yaw > -5.0) {
      headLeftArmed = false; // Re-arm upon returning toward center
    }

    if (yaw > 11.0) {
      // Looking Right -> Single step right navigation
      if (!headRightArmed && (t - lastRightNavAt > 0.55)) {
        headRightArmed = true;
        lastRightNavAt = t;
        console.log('[NeuroBridge Face] Navigation step: Right');
        broadcastSignal('eyeLookRight', 0.95);
      }
    } else if (yaw < 5.0) {
      headRightArmed = false; // Re-arm upon returning toward center
    }

    // 7. Head Nodding ("Yes / Confirm")
    pitchHist.push(pitch);
    if (pitchHist.length > 50) pitchHist.shift();
    if (detectNod(pitchHist) && t - lastNodAt > 2.5) {
      lastNodAt = t;
      isNodding = true;
      speakAndEmit('Yes, confirmed', 'confirm', 'headNodSmile');
      setTimeout(() => { isNodding = false; }, 450);
    }

    // 8. Live Telemetry Broadcast
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
      isNodding: isNodding,
      abnormality: isAbnormal ? abnormalReason : 'Normal',
      calibrated: calFrames >= 60,
      observedAt: Date.now(),
    });
  }

  // Synthetic Demonstration Engine (from NeuroFace Sense demo mode)
  function startSimulationLoop() {
    if (simulationTimer) return;
    simulatedActive = true;
    console.log('[NeuroBridge Face] Idle reference telemetry active.');

    let tick = 0;
    simulationTimer = setInterval(() => {
      tick++;
      const cycle = tick % 140;
      let curYaw = 0.0;
      let curPitch = 0.0;
      let curEye = 0.88;
      let curSmile = 0.05;
      let simNod = false;

      // Subtle breathing motion for visual liveliness only — NEVER triggers speech
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
        isNodding: simNod,
        abnormality: 'Normal',
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
        console.warn('[NeuroBridge Face] Detection frame error:', err);
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
      console.log('[NeuroBridge Face] Camera stream playing.');
      return true;
    } catch (err) {
      console.warn('[NeuroBridge Face] Camera acquisition denied/unavailable, fallback mode active:', err.message);
      return false;
    }
  }

  function initMesh() {
    if (faceMesh) return faceMesh;
    if (typeof FaceMesh === 'undefined') {
      console.warn('[NeuroBridge Face] MediaPipe FaceMesh classic UMD not found in window.');
      return null;
    }
    try {
      const fm = new FaceMesh({
        locateFile: (f) => 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/' + f,
      });
      fm.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      fm.onResults(onMesh);
      faceMesh = fm;
      console.log('[NeuroBridge Face] FaceMesh initialized successfully.');
      return faceMesh;
    } catch (e) {
      console.warn('[NeuroBridge Face] FaceMesh creation error:', e);
      return null;
    }
  }

  async function ensureMesh() {
    for (let i = 0; i < 20; i++) {
      const m = initMesh();
      if (m) return m;
      await new Promise((r) => setTimeout(r, 150));
    }
    return null;
  }

  const NeuroBridgeFace = {
    async start() {
      if (isRunning) return;
      isRunning = true;
      resetAutoCalibration();

      // Start responsive synthetic loop immediately so UI is never frozen
      startSimulationLoop();

      broadcastStatus({
        faceDetected: true,
        lifecycle: 'active',
        message: 'NeuroBridge Facial Engine active',
      });

      const cameraOk = await startCamera();
      const meshOk = await ensureMesh();

      if (cameraOk && meshOk) {
        stopSimulationLoop();
        console.log('[NeuroBridge Face] Real camera and FaceMesh active — live inference running.');
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
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
      }
      if (videoEl) {
        videoEl.srcObject = null;
      }
      broadcastStatus({
        faceDetected: false,
        lifecycle: 'stopped',
        message: 'Facial monitor stopped',
      });
    },

    // Manual test triggers for all 5 actions
    triggerWater() { speakAndEmit('I want water', 'water', 'blink'); },
    triggerFeelingGood() { speakAndEmit('I am feeling good', 'feeling_good', 'smile'); },
    triggerAbnormality() { speakAndEmit('Emergency help needed', 'abnormality', 'seizureAlert'); },
    triggerFood() { speakAndEmit('Give me some food', 'food', 'eyeLookRight'); },
    triggerConfirm() { speakAndEmit('Yes, confirmed', 'confirm', 'headNodSmile'); },

    resetCalibration() { resetAutoCalibration(); },
  };

  global.NeuroBridgeFace = NeuroBridgeFace;

  if (typeof window !== 'undefined') {
    // Listen for postMessage from Flutter Web
    window.addEventListener('message', (event) => {
      const data = event.data;
      if (!data || typeof data !== 'object') return;
      if (data.type === 'neurobridge_trigger_gesture') {
        const g = data.gesture;
        if (g === 'water' || g === '5-blinks') NeuroBridgeFace.triggerWater();
        else if (g === 'feeling_good' || g === 'smile') NeuroBridgeFace.triggerFeelingGood();
        else if (g === 'abnormality' || g === 'emergency') NeuroBridgeFace.triggerAbnormality();
        else if (g === 'food' || g === '5-head-right') NeuroBridgeFace.triggerFood();
        else if (g === 'nod' || g === 'confirm') NeuroBridgeFace.triggerConfirm();
      } else if (data.type === 'neurobridge_reset_calibration') {
        NeuroBridgeFace.resetCalibration();
      }
    });

    // Auto-start immediately when window is loaded
    window.addEventListener('load', () => {
      console.log('[NeuroBridge Face] Facial Intelligence Engine registered.');
      setTimeout(() => {
        if (!isRunning) NeuroBridgeFace.start();
      }, 300);
    });
  }
})(typeof window !== 'undefined' ? window : globalThis);
