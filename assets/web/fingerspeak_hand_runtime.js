(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.FingerSpeakHandRuntime = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  function createPresenceTracker(options) {
    const lostGraceMs = Math.max(0, Number(options?.lostGraceMs ?? 350));
    let present = false;
    let lastSeenAt = Number.NEGATIVE_INFINITY;

    function snapshot(changed) {
      return { present, changed, lastSeenAt };
    }

    return {
      seen(nowMs) {
        const now = Number(nowMs);
        const changed = !present;
        present = true;
        lastSeenAt = Number.isFinite(now) ? now : lastSeenAt;
        return snapshot(changed);
      },

      missed(nowMs) {
        const now = Number(nowMs);
        let changed = false;
        if (present && Number.isFinite(now) && now - lastSeenAt >= lostGraceMs) {
          present = false;
          changed = true;
        }
        return snapshot(changed);
      },

      reset() {
        const changed = present;
        present = false;
        lastSeenAt = Number.NEGATIVE_INFINITY;
        return snapshot(changed);
      },

      get present() {
        return present;
      },

      get lastSeenAt() {
        return lastSeenAt;
      },
    };
  }

  function createCalibrationState(options) {
    const presence = createPresenceTracker(options);
    let trackingReady = false;
    let wizardActive = false;
    let lastDetectionAt = Number.NEGATIVE_INFINITY;

    function snapshot(changed = false, accepted = true) {
      const handPresent = presence.present;
      const controlsReady = trackingReady && handPresent && !wizardActive;
      return {
        handPresent,
        trackingReady,
        wizardActive,
        canManualRecord: controlsReady,
        canStartWizard: controlsReady,
        changed,
        accepted,
        lastSeenAt: presence.lastSeenAt,
      };
    }

    return {
      updateDetection(hasLandmarks, nowMs) {
        const now = Number(nowMs);
        if (!Number.isFinite(now) || now < lastDetectionAt) {
          return snapshot(false, false);
        }
        lastDetectionAt = Math.max(lastDetectionAt, now);
        const state = hasLandmarks ? presence.seen(now) : presence.missed(now);
        return snapshot(state.changed, true);
      },

      setTrackingReady(ready) {
        trackingReady = Boolean(ready);
        return snapshot();
      },

      setWizardActive(active) {
        wizardActive = Boolean(active);
        return snapshot();
      },

      resetDetection() {
        lastDetectionAt = Number.NEGATIVE_INFINITY;
        const state = presence.reset();
        return snapshot(state.changed, true);
      },

      snapshot,

      get handPresent() {
        return presence.present;
      },
    };
  }

  function validTimedFrames(timedFrames) {
    if (!Array.isArray(timedFrames)) return [];
    return timedFrames
      .filter((frame) =>
        frame &&
        Number.isFinite(frame.t) &&
        Array.isArray(frame.feat) &&
        frame.feat.length > 0 &&
        frame.feat.every(Number.isFinite),
      )
      .slice()
      .sort((a, b) => a.t - b.t);
  }

  function describeTimedFrames(timedFrames, windowMs) {
    const frames = validTimedFrames(timedFrames);
    const targetWindowMs = Math.max(1, Number(windowMs) || 1);
    if (frames.length === 0) {
      return { frameCount: 0, spanMs: 0, temporalCoverage: 0, maxGapMs: 0 };
    }

    const spanMs = Math.max(0, frames[frames.length - 1].t - frames[0].t);
    let maxGapMs = 0;
    for (let i = 1; i < frames.length; i++) {
      maxGapMs = Math.max(maxGapMs, frames[i].t - frames[i - 1].t);
    }
    return {
      frameCount: frames.length,
      spanMs,
      temporalCoverage: Math.min(1, spanMs / targetWindowMs),
      maxGapMs,
    };
  }

  function resampleSequence(timedFrames, count, windowMs, options) {
    const frames = validTimedFrames(timedFrames);
    const outputCount = Math.max(2, Math.floor(Number(count) || 0));
    const targetWindowMs = Math.max(1, Number(windowMs) || 1);
    const minCoverage = Math.min(
      1,
      Math.max(0, Number(options?.minCoverage ?? 0.55)),
    );
    if (frames.length < 2) return null;

    const latest = frames[frames.length - 1].t;
    const desiredStart = latest - targetWindowMs;
    let firstIndex = frames.findIndex((frame) => frame.t >= desiredStart);
    if (firstIndex < 0) firstIndex = frames.length - 1;
    // Retain one earlier point so the requested start can be interpolated rather
    // than extrapolated. This matters when MediaPipe runs at a low Android FPS.
    if (firstIndex > 0) firstIndex -= 1;
    const windowFrames = frames.slice(firstIndex);
    if (windowFrames.length < 2) return null;

    const first = windowFrames[0].t;
    const sampleStart = Math.max(desiredStart, first);
    const availableSpan = latest - sampleStart;
    if (availableSpan < targetWindowMs * minCoverage) return null;

    const featureLength = windowFrames[0].feat.length;
    if (windowFrames.some((frame) => frame.feat.length !== featureLength)) return null;

    const out = [];
    let upperIndex = 1;
    for (let i = 0; i < outputCount; i++) {
      const target = sampleStart + (i / (outputCount - 1)) * availableSpan;
      while (
        upperIndex < windowFrames.length - 1 &&
        windowFrames[upperIndex].t < target
      ) {
        upperIndex += 1;
      }
      const hi = windowFrames[upperIndex];
      const lo = windowFrames[Math.max(0, upperIndex - 1)];
      const span = hi.t - lo.t;
      const alpha = span > 0
        ? Math.min(1, Math.max(0, (target - lo.t) / span))
        : 0;
      out.push(lo.feat.map((value, index) =>
        value + (hi.feat[index] - value) * alpha,
      ));
    }
    return out;
  }

  function createStorageAdapter(options) {
    const nativeCall = typeof options?.nativeCall === 'function'
      ? options.nativeCall
      : null;
    const browserStorage = options?.browserStorage ?? null;

    return {
      async set(key, value) {
        const storageKey = String(key);
        const storageValue = String(value);
        if (nativeCall) {
          try {
            const result = await nativeCall({
              action: 'set',
              key: storageKey,
              value: storageValue,
            });
            if (result?.ok) return true;
            if (result?.ok === false) return false;
          } catch (_) {
            // Fall through to browser storage when the native bridge is absent.
          }
        }
        try {
          browserStorage?.setItem(storageKey, storageValue);
          return browserStorage != null;
        } catch (_) {
          return false;
        }
      },

      async get(key) {
        const storageKey = String(key);
        if (nativeCall) {
          try {
            const result = await nativeCall({ action: 'get', key: storageKey });
            if (result?.ok) {
              return result.value == null ? null : { value: String(result.value) };
            }
            if (result?.ok === false) return null;
          } catch (_) {
            // Fall through to browser storage when the native bridge is absent.
          }
        }
        try {
          const value = browserStorage?.getItem(storageKey);
          return value == null ? null : { value };
        } catch (_) {
          return null;
        }
      },
    };
  }

  function createResearchTelemetry(options) {
    const enabled = options?.enabled === true;
    const send = typeof options?.send === 'function' ? options.send : null;
    const now = typeof options?.now === 'function' ? options.now : Date.now;
    const sessionId = String(options?.sessionId || 'hand-studio');
    let sequence = 0;

    return {
      emit(event, fields) {
        if (!enabled || !send || typeof event !== 'string' || !event.trim()) {
          return false;
        }
        const timestamp = Number(now());
        const payload = {
          schema_version: 1,
          event: event.trim(),
          timestamp_ms: Number.isFinite(timestamp) ? Math.round(timestamp) : Date.now(),
          session_id: sessionId,
          sequence: ++sequence,
          ...(fields && typeof fields === 'object' ? fields : {}),
        };
        // Core fields cannot be overridden by event-specific data.
        payload.schema_version = 1;
        payload.event = event.trim();
        payload.timestamp_ms = Number.isFinite(timestamp)
          ? Math.round(timestamp)
          : Date.now();
        payload.session_id = sessionId;
        payload.sequence = sequence;
        return send({ type: 'research_event', payload }) !== false;
      },

      get enabled() {
        return enabled;
      },
    };
  }

  return {
    createCalibrationState,
    createResearchTelemetry,
    createStorageAdapter,
    createPresenceTracker,
    describeTimedFrames,
    resampleSequence,
  };
});
