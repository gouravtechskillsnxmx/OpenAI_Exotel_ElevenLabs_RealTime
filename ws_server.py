# ws_server.py — Exotel-only Realtime Voice Bot (FastAPI + OpenAI Realtime, PCM16)
# ------------------------------------------------------------------------------
# What it does:
# - WebSocket endpoint /exotel-media that Exotel Voicebot Applet connects to
# - Streams caller audio (PCM16) to OpenAI Realtime
# - Streams OpenAI audio (PCM16) back to caller in real time
# - Accumulates ~120ms of audio before each commit to satisfy Realtime API
# - Forces English responses
#
# How to wire in Exotel:
# - Create a Voicebot (bidirectional) applet, set URL to wss://<your-host>/exotel-media
#   OR set it to https://<your-host>/exotel-ws-bootstrap (this returns {"url":"wss://.../exotel-media"})
#
# Env required:
# - One of: OPENAI_KEY / OpenAI_Key / OPENAI_API_KEY
# - PUBLIC_BASE_URL (for /exotel-ws-bootstrap convenience)
#
# Dependencies:
#   pip install fastapi uvicorn aiohttp python-dotenv
#
# Run locally:
#   uvicorn ws_server:app --host 0.0.0.0 --port 10000

import os
import asyncio
import json
import logging
import base64
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from aiohttp import ClientSession, WSMsgType

# ---------- Logging ----------
level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, level, logging.INFO))
logger = logging.getLogger("ws_server")
app = FastAPI()
# ---------- Env ----------
try:
    from dotenv import load_dotenv  # optional for local dev
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = (
    os.getenv("OPENAI_KEY")
    or os.getenv("OpenAI_Key")
    or os.getenv("OPENAI_API_KEY")
)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

if not OPENAI_API_KEY:
    logger.warning("No OpenAI key found. Set OPENAI_KEY or OpenAI_Key or OPENAI_API_KEY.")
if not PUBLIC_BASE_URL:
    logger.info("PUBLIC_BASE_URL not set (only needed for /exotel-ws-bootstrap).")

# ---------- FastAPI ----------
# ---- add to your imports ----
import base64, asyncio, json, os, logging
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from aiohttp import ClientSession, WSMsgType

logger = logging.getLogger("ws_server")

REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")

# ---- add this new route ----
@app.websocket("/browser-media")
async def browser_media_ws(ws: WebSocket):
    """
    Browser softphone WS (no input_audio_buffer.*):
      - Lazy-connects to OpenAI after first non-zero frame
      - For each user turn, sends response.create with inline input_audio (base64 PCM16 chunks)
      - Hard barge-in: response.cancel + immediate new response.create with buffered audio
      - Never calls input_audio_buffer.append/commit → cannot trigger commit_empty
    """
    await ws.accept()
    logger.info("/browser-media connected")

    if not OPENAI_API_KEY:
        logger.error("No OPENAI_API_KEY; closing /browser-media")
        await ws.close()
        return

    # ---- stream state ----
    sr = 16000
    target_sr = 24000
    BYTES_PER_SAMPLE = 2
    MIN_TIME_S = 0.15  # Safe >100ms
    MIN_WINDOW = int(sr * BYTES_PER_SAMPLE * MIN_TIME_S)  # Adjusted

    # Accumulators for the next user turn (when model idle)
    live_chunks: list[str] = []
    live_bytes = 0
    live_frames = 0

    # Accumulators while bot is speaking (barge-in)
    barge_chunks: list[str] = []
    barge_bytes = 0
    barge_frames = 0

    pending = False   # True while a response is in-flight / bot speaking
    speaking = False  # set True while we receive audio deltas

    # OpenAI (lazy)
    openai_session: Optional[ClientSession] = None
    openai_ws = None
    pump_task: Optional[asyncio.Task] = None
    connected_to_openai = False

    async def send_openai(payload: dict):
        if openai_ws is None or openai_ws.closed:
            logger.info("drop %s: OpenAI ws not ready/closed", payload.get("type"))
            return
        t = payload.get("type")
        if t != "response.audio.delta":
            logger.info("SENDING to OpenAI: %s", t)

        await openai_ws.send_json(payload)

    async def openai_connect():
        nonlocal openai_session, openai_ws, pump_task, connected_to_openai, speaking, pending
        if connected_to_openai:
            return

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1"}
        url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

        openai_session = ClientSession()
        openai_ws = await openai_session.ws_connect(url, headers=headers)

        await send_openai({
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 600
                },
                "voice": "verse",
                "instructions": (
                    "You are a concise helpful voice agent. "
                    "Always respond in English (Indian English). Keep answers short."
                ),
            }
        })

        async def pump_to_browser():
            nonlocal speaking, pending
            try:
                async for msg in openai_ws:
                    if msg.type == WSMsgType.TEXT:
                        evt = msg.json()
                        et = evt.get("type")

                        if et in ("response.output_audio.delta", "response.audio.delta"):
                            chunk = evt.get("delta")
                            if chunk and ws.client_state.name != "DISCONNECTED":
                                speaking = True
                                await ws.send_text(json.dumps({"event": "media", "audio": chunk}))

                        elif et == "response.completed":
                            speaking = False
                            pending = False
                            logger.info("OpenAI: response.completed (pending -> False)")

                        elif et == "error":
                            logger.error("OpenAI error event: %s", evt)
                            pending = False
                            break

                    elif msg.type == WSMsgType.ERROR:
                        logger.error("OpenAI ws error")
                        pending = False
                        break
            except Exception as e:
                logger.exception("OpenAI pump error: %s", e)
                pending = False

        pump_task = asyncio.create_task(pump_to_browser())
        connected_to_openai = True
        logger.info("OpenAI realtime connected (lazy)")

    async def openai_close():
        nonlocal connected_to_openai
        try:
            if pump_task and not pump_task.done():
                pump_task.cancel()
        except Exception:
            pass
        try:
            if openai_ws and not openai_ws.closed:
                await openai_ws.close()
        except Exception:
            pass
        try:
            if openai_session:
                await openai_session.close()
        except Exception:
            pass
        connected_to_openai = False

    async def send_turn_from_chunks(chunks: list[str]):
        """Resample whole turn, append, commit, then request response."""
        nonlocal pending

        if not chunks:
            return

        # Concat samples from chunks
        samples_list = []
        for c in chunks:
            audio_bytes = base64.b64decode(c)
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            samples_list.append(samples)
        all_samples = np.concatenate(samples_list)

        # Resample if needed
        if sr != target_sr:
            target_num = int(len(all_samples) * (target_sr / sr))
            if target_num == 0:
                logger.info("Skip commit: resampled to 0 samples")
                return
            resampled = resample(all_samples, target_num)
            resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
        else:
            resampled = all_samples.astype(np.int16)

        # Check min duration after resample (should match input time)
        resampled_ms = (len(resampled) / target_sr) * 1000
        if resampled_ms < (MIN_TIME_S * 1000):
            logger.info("Skip commit: resampled %.2fms < %.2fms", resampled_ms, MIN_TIME_S * 1000)
            return

        resampled_b64 = base64.b64encode(resampled.tobytes()).decode('utf-8')

        # Append
        await send_openai({
            "type": "input_audio_buffer.append",
            "audio": resampled_b64
        })

        # Commit
        await send_openai({"type": "input_audio_buffer.commit"})

        # Request response
        await send_openai({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": "Reply in English only. Keep it short."
            }
        })

        pending = True
        logger.info("turn sent: chunks=%d resampled_ms=%.2f", len(chunks), resampled_ms)

    try:
        while True:
            raw = await ws.receive_text()
            m = json.loads(raw)
            ev = m.get("event")

            if ev == "start":
                try:
                    sr = int(m.get("sample_rate") or 16000)
                except Exception:
                    sr = 16000
                MIN_WINDOW = int(sr * BYTES_PER_SAMPLE * MIN_TIME_S)
                live_chunks.clear(); live_bytes = live_frames = 0
                barge_chunks.clear(); barge_bytes = barge_frames = 0
                pending = speaking = False
                logger.info("/browser-media start sr=%d min_window=%d", sr, MIN_WINDOW)

            elif ev == "media":
                b64 = m.get("audio")
                if not b64:
                    logger.info("drop frame: empty base64")
                    continue

                # Validate bytes
                try:
                    blen = len(base64.b64decode(b64))
                except Exception:
                    blen = 0
                if blen == 0:
                    logger.info("frame bytes=0 (ignored)")
                    continue

                # First real frame → connect now
                if not connected_to_openai:
                    await openai_connect()

                # If model is speaking, accumulate for barge-in
                if pending or speaking:
                    barge_chunks.append(b64)
                    barge_bytes += blen
                    barge_frames += 1
                    logger.info("buffering while pending: +%d (total=%d, frames=%d)",
                                blen, barge_bytes, barge_frames)

                    # Once we have enough, cancel and send new turn
                    if barge_bytes >= MIN_WINDOW and barge_frames >= 2:
                        logger.info("barge-in: cancel current TTS and send new turn "
                                    "(frames=%d bytes=%d)", barge_frames, barge_bytes)
                        await send_openai({"type": "response.cancel"})
                        speaking = False
                        pending = False
                        await asyncio.sleep(0)  # Yield for cancel
                        await send_turn_from_chunks(barge_chunks)
                        barge_chunks.clear(); barge_bytes = barge_frames = 0
                    continue

                # Model idle: accumulate live window
                live_chunks.append(b64)
                live_bytes += blen
                live_frames += 1
                logger.info("frame bytes=%d, live_bytes=%d, live_frames=%d, need>=%d",
                            blen, live_bytes, live_frames, MIN_WINDOW)

                if live_bytes >= MIN_WINDOW and live_frames >= 2 and not pending:
                    await send_turn_from_chunks(live_chunks)
                    live_chunks.clear(); live_bytes = live_frames = 0

            else:
                # ignore unknown events
                pass

    except WebSocketDisconnect:
        logger.info("/browser-media disconnected")
    except Exception as e:
        logger.exception("/browser-media error: %s", e)
    finally:
        await openai_close()
        try:
            await ws.close()
        except Exception:
            pass
#------------------------------------------------------------



# ---------- Health / Diag ----------
@app.get("/health")
async def health():
    return PlainTextResponse("ok", status_code=200)

@app.get("/diag")
async def diag():
    return {
        "openai_key_present": bool(OPENAI_API_KEY),
        "public_base_url_set": bool(PUBLIC_BASE_URL),
    }

# ---------- Exotel WS bootstrap ----------
# If you prefer to give Exotel an HTTPS endpoint that returns the WS URL:
@app.get("/exotel-ws-bootstrap")
async def exotel_ws_bootstrap():
    if not PUBLIC_BASE_URL:
        return JSONResponse({"error": "PUBLIC_BASE_URL not configured"}, status_code=500)
    return {"url": f"wss://{PUBLIC_BASE_URL.split('://')[-1]}/exotel-media"}

# ======================================================================
# ===============  EXOTEL BIDIRECTIONAL WS HANDLER  ====================
# ======================================================================
# Exotel will send JSON events:
#   "connected" (optional)
#   "start"  { start: { stream_sid, media_format: { encoding, sample_rate, bit_rate } } }
#   "media"  { media: { payload: "<base64 of PCM16 mono>" } }
#   "dtmf"   ...
#   "stop"
#
# We reply with:
#   {"event":"media","stream_sid": "...", "media":{"payload":"<base64 PCM16>"}}
#
# OpenAI Realtime expects:
#   session.update: input/output audio format strings ("pcm16")
#   input_audio_buffer.append: { audio: "<base64 PCM16>" }
#   input_audio_buffer.commit  (after >= ~100ms buffered)
#   response.create            (modalities ["text","audio"])
#
# Notes:
# - We accumulate ~120ms of audio (commit_target) based on incoming sample_rate.
# - Optional "hard barge-in" is included (commented out). Enable if you want to force-cut current speech.

REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")

@app.websocket("/exotel-media")
async def exotel_media_ws(ws: WebSocket):
    await ws.accept()
    logger.info("Exotel WS connected")

    if not OPENAI_API_KEY:
        logger.error("No OPENAI_API_KEY; closing Exotel stream.")
        await ws.close()
        return

    # Stream state
    stream_sid: Optional[str] = None
    sample_rate: int = 8000  # default; will be updated from "start"
    bytes_per_sample: int = 2  # PCM16 mono
    commit_target: int = int(sample_rate * bytes_per_sample * 0.12)  # ~120ms
    accum_bytes: int = 0
    speaking: bool = False  # for optional hard barge-in

    openai_session: Optional[ClientSession] = None
    openai_ws = None
    openai_reader_task: Optional[asyncio.Task] = None

    async def openai_connect():
        """Open the Realtime WS to OpenAI and configure the session for PCM16 + English."""
        nonlocal openai_session, openai_ws, openai_reader_task, speaking
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1"}
        url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

        openai_session = ClientSession()
        openai_ws = await openai_session.ws_connect(url, headers=headers)

        # Configure session once (PCM16 both ways, English-only instructions)
        await openai_ws.send_json({
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 600
                },
                "voice": "verse",
                "instructions": (
                    "You are a concise helpful voice agent. "
                    "Always respond in English (Indian English). Keep answers short."
                ),
            }
        })

        # Start a background task to forward OpenAI audio deltas back to Exotel
        async def pump_openai_to_exotel():
            nonlocal speaking
            try:
                async for msg in openai_ws:
                    if msg.type == WSMsgType.TEXT:
                        evt = msg.json()
                        etype = evt.get("type")
                        if etype == "response.audio.delta":
                            chunk_b64 = evt.get("delta")
                            if chunk_b64 and ws.client_state.name != "DISCONNECTED":
                                speaking = True
                                await ws.send_text(json.dumps({
                                    "event": "media",
                                    "stream_sid": stream_sid,
                                    "media": {"payload": chunk_b64}
                                }))
                        elif etype == "response.completed":
                            speaking = False
                        elif etype == "error":
                            logger.error("OpenAI error: %s", evt)
                            break
                    elif msg.type == WSMsgType.ERROR:
                        logger.error("OpenAI ws error")
                        break
            except Exception as e:
                logger.exception("OpenAI pump error: %s", e)

        openai_reader_task = asyncio.create_task(pump_openai_to_exotel())

    async def openai_close():
        """Gracefully close OpenAI WS and session."""
        try:
            if openai_reader_task and not openai_reader_task.done():
                openai_reader_task.cancel()
        except Exception:
            pass
        try:
            if openai_ws and not openai_ws.closed:
                await openai_ws.close()
        except Exception:
            pass
        try:
            if openai_session:
                await openai_session.close()
        except Exception:
            pass

    # Connect to OpenAI once we have a client WS
    await openai_connect()

    try:
        while True:
            raw = await ws.receive_text()
            evt = json.loads(raw)
            etype = evt.get("event")

            if etype == "connected":
                # Optional first event
                continue

            if etype == "start":
                # Exotel start event; also carries media_format
                start_obj = evt.get("start", {})
                stream_sid = start_obj.get("stream_sid") or start_obj.get("streamSid")
                mf = start_obj.get("media_format") or {}
                sr = int(mf.get("sample_rate") or sample_rate)
                sample_rate = sr
                commit_target = int(sample_rate * bytes_per_sample * 0.12)  # ~120ms buffer
                accum_bytes = 0
                logger.info("Exotel stream started sid=%s sr=%d commit_target=%d", stream_sid, sample_rate, commit_target)

            elif etype == "media":
                # Incoming PCM16 mono base64
                media = evt.get("media") or {}
                payload_b64 = media.get("payload")
                if not payload_b64:
                    continue

                if openai_ws is None or openai_ws.closed:
                    logger.warning("OpenAI WS not ready; skipping audio frame")
                    continue

                # OPTIONAL HARD BARGE-IN:
                # If user speaks while bot is speaking, force-cancel current response
                # if speaking:
                #     await openai_ws.send_json({"type": "response.cancel"})
                #     speaking = False

                # Append audio to OpenAI input buffer
                await openai_ws.send_json({
                    "type": "input_audio_buffer.append",
                    "audio": payload_b64
                })

                # Count bytes and commit when reaching ~120ms
                try:
                    raw_len = len(base64.b64decode(payload_b64))
                except Exception:
                    raw_len = 0
                accum_bytes += raw_len

                if accum_bytes >= commit_target:
                    await openai_ws.send_json({"type": "input_audio_buffer.commit"})
                    await openai_ws.send_json({
                        "type": "response.create",
                        "response": {
                            "modalities": ["text", "audio"],
                            "instructions": "Reply in English only. Keep it short."
                        }
                    })
                    logger.debug("[%s] committed %d bytes; requested response", stream_sid, accum_bytes)
                    accum_bytes = 0

            elif etype == "dtmf":
                # Optional: handle DTMF here if needed
                pass

            elif etype == "stop":
                logger.info("Exotel stream stopped sid=%s", stream_sid)
                break

            # Ignore unknown events quietly
    except WebSocketDisconnect:
        logger.info("Exotel WS disconnected")
    except Exception as e:
        logger.exception("Exotel WS error: %s", e)
    finally:
        await openai_close()
        try:
            await ws.close()
        except Exception:
            pass
