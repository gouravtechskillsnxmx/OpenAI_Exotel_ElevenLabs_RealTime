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
    await ws.accept()
    logger.info("/browser-media connected")

    if not OPENAI_API_KEY:
        logger.error("No OPENAI_API_KEY; closing /browser-media")
        await ws.close()
        return

    # ---- stream state ----
    sr = 16000                    # browser announces in first "start"
    BYTES_PER_SAMPLE = 2          # PCM16 mono
    target = int(sr * BYTES_PER_SAMPLE * 0.12)   # ~120ms
    accum = 0                     # bytes accumulated since last commit
    real_frames = 0               # count of non-zero frames
    pending = False               # waiting on an OpenAI response
    speaking = False              # model is currently speaking

    openai_session: Optional[ClientSession] = None
    openai_ws = None
    pump_task: Optional[asyncio.Task] = None

    async def safe_send_json(payload: dict):
        """
        Always use this instead of openai_ws.send_json.
        It logs the type and blocks empty commits by checking our local buffer state.
        """
        nonlocal accum, real_frames, pending
        t = payload.get("type")
        if t == "input_audio_buffer.commit":
            # HARD GUARD: never allow an empty commit out
            if real_frames < 1 or accum <= 0:
                logger.debug(
                    "BLOCKED commit: accum=%d real_frames=%d pending=%s",
                    accum, real_frames, pending
                )
                return
            logger.debug(
                "SENDING commit: accum=%d real_frames=%d pending=%s",
                accum, real_frames, pending
            )
        elif t == "input_audio_buffer.append":
            # keep logs concise; append is spammy, we log size separately elsewhere
            pass
        else:
            logger.debug("SENDING to OpenAI: %s", t)

        if openai_ws is None or openai_ws.closed:
            logger.debug("OpenAI ws not ready/closed; drop %s", t)
            return
        await openai_ws.send_json(payload)

    async def openai_connect():
        nonlocal openai_session, openai_ws, pump_task, speaking, pending
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1"}
        url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

        openai_session = ClientSession()
        openai_ws = await openai_session.ws_connect(url, headers=headers)

        await safe_send_json({
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

        async def pump_openai_to_browser():
            nonlocal speaking, pending
            try:
                async for msg in openai_ws:
                    if msg.type == WSMsgType.TEXT:
                        evt = msg.json()
                        et = evt.get("type")
                        if et == "response.audio.delta":
                            chunk = evt.get("delta")
                            if chunk and ws.client_state.name != "DISCONNECTED":
                                speaking = True
                                await ws.send_text(json.dumps({"event": "media", "audio": chunk}))
                        elif et == "response.completed":
                            speaking = False
                            pending = False
                            logger.debug("OpenAI: response.completed (pending -> False)")
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

        pump_task = asyncio.create_task(pump_openai_to_browser())

    async def openai_close():
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

    await openai_connect()

    try:
        while True:
            raw = await ws.receive_text()
            m = json.loads(raw)
            ev = m.get("event")

            if ev == "start":
                # Reset counters and clear any remote buffer (defensive)
                try:
                    sr = int(m.get("sample_rate") or 16000)
                except Exception:
                    sr = 16000
                target = int(sr * BYTES_PER_SAMPLE * 0.12)
                accum = 0
                real_frames = 0
                pending = False
                logger.info("/browser-media start sr=%d target=%d", sr, target)
                await safe_send_json({"type": "input_audio_buffer.clear"})
                logger.debug("remote input buffer cleared")

            elif ev == "media":
                if openai_ws is None or openai_ws.closed:
                    logger.debug("drop frame: OpenAI ws not ready/closed")
                    continue

                b64 = m.get("audio")
                if not b64:
                    logger.debug("drop frame: empty base64")
                    continue

                # Measure first; append only if >0
                try:
                    blen = len(base64.b64decode(b64))
                except Exception:
                    blen = 0

                if blen == 0:
                    logger.debug("frame bytes=0 (ignored)")
                    continue

                # OPTIONAL hard barge-in
                # if speaking:
                #     await safe_send_json({"type": "response.cancel"})
                #     speaking = False
                #     logger.debug("barge-in: response.cancel sent")

                # Append & update counters
                await safe_send_json({"type": "input_audio_buffer.append", "audio": b64})
                accum += blen
                real_frames += 1
                logger.debug("frame bytes=%d, accum=%d, frames=%d, target=%d, pending=%s",
                             blen, accum, real_frames, target, pending)

                # Strict commit gate: >=2 frames, >=target, not pending
                if (real_frames >= 2) and (accum >= target) and (not pending):
                    pending = True
                    await safe_send_json({"type": "input_audio_buffer.commit"})
                    await safe_send_json({
                        "type": "response.create",
                        "response": {
                            "modalities": ["text", "audio"],
                            "instructions": "Reply in English only. Keep it short."
                        }
                    })
                    logger.debug("commit sent (accum was %d). Resetting window.", accum)
                    accum = 0
                    real_frames = 0

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
#------------------------------------------------------------------------



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
