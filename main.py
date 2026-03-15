# main.py
import os
import uuid
import logging
import asyncio
import ffmpeg
import shutil
import assemblyai as aai
import requests
from typing import Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from dotenv import load_dotenv
import platform
import subprocess
import multiprocessing

load_dotenv()
num_cores = multiprocessing.cpu_count()

# -----------------------------------------------------
# Logging setup
# -----------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# Directories
# -----------------------------------------------------
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
TEMP_DIR   = os.path.join(BASE_DIR, "temp")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
FONTS_DIR  = os.path.join(BASE_DIR, "fonts")
AUDIO_DIR  = os.path.join(BASE_DIR, "audio")

for d in [TEMP_DIR, VIDEOS_DIR, FONTS_DIR, AUDIO_DIR]:
    os.makedirs(d, exist_ok=True)

if platform.system() != "Windows":
    subprocess.run(["fc-cache", "-fv", FONTS_DIR])
os.environ["FONTCONFIG_PATH"] = FONTS_DIR

# -----------------------------------------------------
# FastAPI setup
# -----------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/videos", StaticFiles(directory=VIDEOS_DIR), name="videos")
app.mount("/audio",  StaticFiles(directory=AUDIO_DIR),  name="audio")

# -----------------------------------------------------
# Globals
# -----------------------------------------------------
uploads: Dict[str, Dict] = {}
executor = ThreadPoolExecutor(max_workers=2)

# -----------------------------------------------------
# AssemblyAI Configuration
# -----------------------------------------------------
# Compatible with both old SDK (<= 0.26) and new SDK (>= 0.27).
# New SDK auto-reads ASSEMBLYAI_API_KEY from environment.
# Old SDK requires explicit assignment to aai.settings.api_key.
_AAI_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
try:
    aai.settings.api_key = _AAI_KEY
except Exception:
    pass   # newer SDK reads the env var automatically
logger.info("Using AssemblyAI for transcription (Hindi mode).")

# =====================================================
# edge_tts Voice catalogue
# Taken directly from your working script configuration.
# =====================================================
#
# Hindi voices  (hi-IN-SwaraNeural, hi-IN-MadhurNeural)
# English voices (en-US-JennyNeural, en-US-GuyNeural,
#                 en-GB-SoniaNeural, en-GB-RyanNeural)
#
# Parameters match your script exactly:
#   rate   : speed adjustment  e.g. "+0%", "+20%", "-10%"
#   volume : loudness           e.g. "+50%", "+0%"
#   pitch  : pitch adjustment   e.g. "+5Hz", "+0Hz", "-5Hz"
# =====================================================

TTS_VOICES = [
    # ── Hindi ──────────────────────────────────────────────
    {
        "voice_id": "hi_female",
        "edge_voice": "hi-IN-SwaraNeural",
        "lang":    "hi",
        "gender":  "Female",
        "name":    "Swara",
        "sub":     "Hindi · Female · Sweet & Natural ✨",
        "rate":    "+0%",
        "volume":  "+50%",
        "pitch":   "+5Hz",
    },
    {
        "voice_id": "hi_male",
        "edge_voice": "hi-IN-MadhurNeural",
        "lang":    "hi",
        "gender":  "Male",
        "name":    "Madhur",
        "sub":     "Hindi · Male · Warm & Clear",
        "rate":    "+0%",
        "volume":  "+50%",
        "pitch":   "+0Hz",
    },
    # ── English US ─────────────────────────────────────────
    {
        "voice_id": "en_us_female",
        "edge_voice": "en-US-JennyNeural",
        "lang":    "en",
        "gender":  "Female",
        "name":    "Jenny",
        "sub":     "US English · Female · Friendly",
        "rate":    "+0%",
        "volume":  "+50%",
        "pitch":   "+0Hz",
    },
    {
        "voice_id": "en_us_male",
        "edge_voice": "en-US-GuyNeural",
        "lang":    "en",
        "gender":  "Male",
        "name":    "Guy",
        "sub":     "US English · Male · Neutral",
        "rate":    "+0%",
        "volume":  "+50%",
        "pitch":   "+0Hz",
    },
    # ── English GB ─────────────────────────────────────────
    {
        "voice_id": "en_gb_female",
        "edge_voice": "en-GB-SoniaNeural",
        "lang":    "en",
        "gender":  "Female",
        "name":    "Sonia",
        "sub":     "British English · Female · Natural",
        "rate":    "+0%",
        "volume":  "+50%",
        "pitch":   "+0Hz",
    },
    {
        "voice_id": "en_gb_male",
        "edge_voice": "en-GB-RyanNeural",
        "lang":    "en",
        "gender":  "Male",
        "name":    "Ryan",
        "sub":     "British English · Male · Clear",
        "rate":    "+0%",
        "volume":  "+50%",
        "pitch":   "+0Hz",
    },
]

# Quick lookup
VOICE_MAP = {v["voice_id"]: v for v in TTS_VOICES}


# =====================================================
# edge_tts synthesis  (async, with retry + gTTS fallback)
# =====================================================

async def _edge_tts_save(text: str, voice: dict, out_path: str, rate: str, volume: str, pitch: str):
    """
    Core edge_tts call — uses your exact script configuration.
    Retries up to 3 times on 403/connection errors.
    """
    import edge_tts

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice["edge_voice"],
        rate=rate,
        volume=volume,
        pitch=pitch,
    )
    await communicate.save(out_path)


def synthesize_edge_tts(text: str, voice_id: str, rate: str, volume: str, pitch: str) -> str:
    """
    Synthesize using edge_tts with 3 retries.
    Falls back to gTTS if all retries fail (e.g. Microsoft 403).
    Returns absolute path to MP3 in AUDIO_DIR.
    """
    if voice_id not in VOICE_MAP:
        raise ValueError(f"Unknown voice_id '{voice_id}'.")

    voice   = VOICE_MAP[voice_id]
    uid     = uuid.uuid4().hex
    mp3_out = os.path.join(AUDIO_DIR, f"tts_{uid}.mp3")

    logger.info(f"edge_tts: voice={voice['edge_voice']} | rate={rate} | volume={volume} | pitch={pitch} | chars={len(text)}")

    # Try edge_tts with up to 3 retries
    last_error = None
    for attempt in range(1, 4):
        try:
            asyncio.run(_edge_tts_save(text, voice, mp3_out, rate, volume, pitch))
            logger.info(f"edge_tts MP3 saved: {mp3_out}")
            return mp3_out
        except Exception as e:
            last_error = e
            logger.warning(f"edge_tts attempt {attempt}/3 failed: {e}")
            if attempt < 3:
                import time
                time.sleep(2 * attempt)   # wait 2s, 4s before retrying

    # All retries failed — fall back to gTTS
    logger.warning(f"edge_tts failed after 3 attempts ({last_error}). Falling back to gTTS…")
    return _gtts_fallback(text, voice, mp3_out)


def _gtts_fallback(text: str, voice: dict, mp3_out: str) -> str:
    """
    gTTS fallback when edge_tts is unavailable (Microsoft 403).
    Applies pitch shift for male voices to mimic the edge_tts voice gender.
    """
    from gtts import gTTS

    uid     = uuid.uuid4().hex
    raw_mp3 = os.path.join(TEMP_DIR, f"tts_raw_{uid}.mp3")

    gtts_lang = "hi" if voice["lang"] == "hi" else "en"
    logger.info(f"gTTS fallback: lang={gtts_lang} | gender={voice['gender']}")

    gTTS(text=text, lang=gtts_lang, slow=False).save(raw_mp3)

    # Apply pitch shift for male voices
    filters = []
    if voice["gender"] == "Male":
        base_rate = 24000
        ratio     = 2 ** (-3.0 / 12.0)          # -3 semitones for male tone
        filters.append(f"asetrate={int(base_rate * ratio)}")
        filters.append(f"aresample={base_rate}")

    af_str = ",".join(filters) if filters else "anull"

    try:
        (
            ffmpeg
            .input(raw_mp3)
            .output(mp3_out, af=af_str, audio_bitrate="192k", acodec="libmp3lame")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        logger.info(f"gTTS fallback MP3 saved: {mp3_out}")
    finally:
        try:
            os.remove(raw_mp3)
        except OSError:
            pass

    return mp3_out


# =====================================================
# Audio effects post-processor (ffmpeg)
# =====================================================
# Echo is implemented using ffmpeg's aecho filter:
#   aecho=in_gain:out_gain:delays:decays
#   - in_gain  : input signal level  (0.0–1.0)
#   - out_gain : output signal level (0.0–1.0)
#   - delays   : comma-sep delay(s) in ms  e.g. "500|1000"
#   - decays   : comma-sep decay per delay e.g. "0.4|0.2"
#
# Reverb uses aecho with very short multi-tap delays to
# simulate room ambience.
# =====================================================

def apply_audio_effects(
    mp3_in: str,
    mp3_out: str,
    echo_enabled:  bool  = False,
    echo_delay:    int   = 500,    # ms  (50 – 2000)
    echo_decay:    float = 0.4,    # 0.1 – 0.9
    echo_taps:     int   = 1,      # 1 = single echo, 2-3 = multiple repeats
    reverb_enabled: bool = False,
    reverb_amount:  float = 0.3,   # 0.1 – 1.0 (room size feel)
) -> str:
    """
    Apply echo and/or reverb to an MP3 using ffmpeg aecho filter.
    Returns mp3_out path (same file as input if no effects active).
    """
    if not echo_enabled and not reverb_enabled:
        # No effects — just copy input to output path
        import shutil as _sh
        _sh.copy2(mp3_in, mp3_out)
        return mp3_out

    filters = []

    if reverb_enabled:
        # Reverb = dense short multi-tap echo (room ambience simulation)
        amt = max(0.05, min(1.0, float(reverb_amount)))
        # 4 short taps: 20ms, 40ms, 60ms, 80ms with decaying levels
        delays = "20|40|60|80"
        decays = f"{amt*0.8:.2f}|{amt*0.6:.2f}|{amt*0.4:.2f}|{amt*0.2:.2f}"
        filters.append(f"aecho=0.8:0.9:{delays}:{decays}")

    if echo_enabled:
        # Echo = distinct delayed repeat(s)
        delay = max(50, min(2000, int(echo_delay)))
        decay = max(0.1, min(0.9, float(echo_decay)))
        taps  = max(1, min(4, int(echo_taps)))

        # Build multi-tap delays and decays
        delay_list = "|".join(str(delay * (i + 1)) for i in range(taps))
        decay_list = "|".join(f"{decay * (0.7 ** i):.2f}" for i in range(taps))
        filters.append(f"aecho=0.8:0.88:{delay_list}:{decay_list}")

    af_str = ",".join(filters)

    logger.info(f"Applying audio effects: {af_str}")
    try:
        (
            ffmpeg
            .input(mp3_in)
            .output(mp3_out, af=af_str, audio_bitrate="192k", acodec="libmp3lame")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        logger.info(f"Effects applied → {mp3_out}")
    except ffmpeg.Error as e:
        stderr_msg = e.stderr.decode("utf-8", errors="replace") if e.stderr else "no stderr"
        logger.error(f"ffmpeg effects failed: {stderr_msg}")
        raise RuntimeError(f"ffmpeg effects error: {stderr_msg}") from e
    finally:
        try:
            if mp3_in != mp3_out:
                os.remove(mp3_in)
        except OSError:
            pass

    return mp3_out


# -----------------------------------------------------
# Caption helpers (original — unchanged)
# -----------------------------------------------------
def sec_to_srt(ts: float) -> str:
    h, m = int(ts // 3600), int((ts % 3600) // 60)
    s    = int(ts % 60)
    ms   = int((ts - int(ts)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def save_srt_from_text(srt_text: str, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(srt_text)
    logger.info(f"SRT saved: {filename}")
    return filename


# ── Caption style helpers ────────────────────────────────────

def hex_to_ass_color(hex_color: str, alpha: int = 0) -> str:
    """
    Convert #RRGGBB web hex to ASS &HAABBGGRR format.
    alpha: 0=fully opaque, 255=fully transparent
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


# ASS alignment codes:
#   7=top-left    8=top-center    9=top-right
#   4=mid-left    5=mid-center    6=mid-right
#   1=bot-left    2=bot-center    3=bot-right
POSITION_MAP = {
    "bottom_center": 2,
    "bottom_left":   1,
    "bottom_right":  3,
    "middle_center": 5,
    "middle_left":   4,
    "middle_right":  6,
    "top_center":    8,
    "top_left":      7,
    "top_right":     9,
}


def _srt_time_to_ms(t: str) -> int:
    """Convert SRT timestamp string 'HH:MM:SS,mmm' to milliseconds."""
    t = t.strip()
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)


def convert_srt_to_ass_karaoke(
    srt_file, ass_file,
    font_size: int      = 56,
    text_color: str     = "#FFFFFF",    # web hex — normal word colour
    highlight_color: str = "#FFFF00",  # web hex — word highlight as spoken
    outline_color: str  = "#000000",    # web hex
    bg_color: str       = "#000000",    # web hex (box background)
    bg_alpha: int       = 100,          # 0=opaque, 255=transparent
    position: str       = "bottom_center",
    bold: bool          = True,
    words_json: list    = None,         # word-level timestamps from AssemblyAI
):
    """
    Build an ASS subtitle file with karaoke word highlighting.

    If words_json is provided (list of {text, start, end} in ms),
    each word gets an exact \\kf tag based on its real spoken duration —
    the word smoothly transitions from text_color to highlight_color
    as it is being spoken.

    If words_json is None, falls back to even distribution across the subtitle.
    """
    import json as _json

    try:
        logger.info(f"SRT->ASS | size={font_size} textColor={text_color} hlColor={highlight_color} pos={position}")

        srt_content = ""
        with open(srt_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                srt_content += line
        srt_content = srt_content.strip()

        mukta_path = os.path.join(FONTS_DIR, "NotoSansDevanagari-Regular.ttf")
        fontname   = "NotoSansDevanagari" if os.path.exists(mukta_path) else "Mukta"

        primary_col   = hex_to_ass_color(text_color,      alpha=0)
        secondary_col = hex_to_ass_color(highlight_color, alpha=0)   # \k fill colour
        outline_col   = hex_to_ass_color(outline_color,   alpha=0)
        back_col      = hex_to_ass_color(bg_color,        alpha=bg_alpha)
        alignment     = POSITION_MAP.get(position, 2)
        bold_flag     = 1 if bold else 0

        # SecondaryColour in ASS is the karaoke fill colour (the highlight)
        ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes
Collisions: Normal
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},{font_size},{primary_col},{secondary_col},{outline_col},{back_col},{bold_flag},0,0,0,100,100,0,0,1,3,2,{alignment},30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        # Build a word-timestamp lookup: normalised_text → list of (start_ms, end_ms)
        # We pop from the front so repeated words work correctly.
        from collections import deque
        word_lookup: dict = {}
        if words_json:
            for w in words_json:
                key = w["text"].strip().lower()
                if key not in word_lookup:
                    word_lookup[key] = deque()
                word_lookup[key].append((int(w["start"]), int(w["end"])))

        srt_content = srt_content.replace("\r\n", "\n").replace("\r", "\n")
        blocks      = srt_content.split("\n\n")
        ass_lines   = []

        def srt_to_ass_time(t: str) -> str:
            t = t.strip()
            body, ms = t.split(",")
            h, m, s  = body.split(":")
            return f"{h}:{m}:{s}.{ms[:2]}"

        for block in blocks:
            lines = block.split("\n")
            if len(lines) < 3:
                continue
            time_line = lines[1].strip()
            text      = " ".join(lines[2:]).strip()
            if not text or " --> " not in time_line:
                continue

            start_str, end_str = time_line.split(" --> ")
            ass_start = srt_to_ass_time(start_str)
            ass_end   = srt_to_ass_time(end_str)

            block_start_ms = _srt_time_to_ms(start_str)
            block_end_ms   = _srt_time_to_ms(end_str)
            block_dur_ms   = max(block_end_ms - block_start_ms, 1)

            word_tokens = text.split()

            # Build karaoke line with \kf tags (fill-sweep highlight)
            kar_parts = []
            for w in word_tokens:
                key = w.strip().lower().rstrip('.,!?;:\'"')
                q   = word_lookup.get(key)
                if q:
                    w_start, w_end = q[0]
                    # Only consume the timestamp if it falls inside this subtitle block
                    if block_start_ms - 200 <= w_start <= block_end_ms + 200:
                        q.popleft()
                        if not q:
                            del word_lookup[key]
                        dur_cs = max(int((w_end - w_start) / 10), 1)  # centiseconds
                    else:
                        # Word timestamp is outside this block — fall back to even split
                        dur_cs = max(int(block_dur_ms / max(len(word_tokens), 1) / 10), 1)
                else:
                    # No timestamp found — even distribution fallback
                    dur_cs = max(int(block_dur_ms / max(len(word_tokens), 1) / 10), 1)

                # \\kf = karaoke fill sweep (word lights up from left to right)
                kar_parts.append(f"{{\\kf{dur_cs}}}{w}")

            kar_line = " ".join(kar_parts)
            ass_lines.append(
                f"Dialogue: 0,{ass_start},{ass_end},Default,,0,0,0,,{{\an{alignment}}}{kar_line}"
            )

        with open(ass_file, "w", encoding="utf-8-sig") as f:
            f.write(ass_header + "\n" + "\n".join(ass_lines))

        logger.info(f"🟢ASS karaoke conversion successful: {ass_file}")
        return True

    except Exception as e:
        logger.exception("🔴ASS conversion failed")
        return False


def _safe_ass_path(p: str) -> str:
    """Windows-safe path for ffmpeg filter string."""
    p = p.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\:" + p[2:]
    return p


def burn_subtitles(
    video_path, srt_file, output_path,
    font_size: int       = 56,
    text_color: str      = "#FFFFFF",
    highlight_color: str = "#FFFF00",
    outline_color: str   = "#000000",
    bg_color: str        = "#000000",
    bg_alpha: int        = 100,
    position: str        = "bottom_center",
    bold: bool           = True,
    words_json: list     = None,
):
    logger.info("🔥Burning subtitles to video...")
    ass_file = srt_file.replace('.srt', '.ass')
    if not convert_srt_to_ass_karaoke(
        srt_file, ass_file,
        font_size=font_size,
        text_color=text_color,
        highlight_color=highlight_color,
        outline_color=outline_color,
        bg_color=bg_color,
        bg_alpha=bg_alpha,
        position=position,
        bold=bold,
        words_json=words_json,
    ):
        raise RuntimeError("🔴Failed to convert SRT to ASS")
    logger.info("🔥ffmpeg.input starting...")
    safe_ass = _safe_ass_path(ass_file)
    try:
        ffmpeg.input(video_path).output(
            output_path,
            vf=f"ass={safe_ass}",
            crf=18,
            preset='ultrafast',
            tune='fastdecode',
            movflags='+faststart',
            vsync='vfr',
            threads=num_cores
        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        stderr_msg = e.stderr.decode("utf-8", errors="replace") if e.stderr else "no stderr"
        logger.error(f"🔴ffmpeg burn failed:\n{stderr_msg}")
        raise RuntimeError(f"ffmpeg error: {stderr_msg}") from e
    logger.info(f"🎞Burn completed: {output_path}")
    try:
        os.remove(ass_file)
    except Exception:
        pass
    return True


# -----------------------------------------------------
# AssemblyAI transcription — returns (srt_text, words_json)
# words_json is a list of {text, start_ms, end_ms} dicts
# used for precise karaoke word-level timing.
# -----------------------------------------------------
def transcribe_with_assemblyai(audio_path: str):
    """
    Returns tuple: (srt_text: str, words: list[dict])
    words = [{"text": "hello", "start": 120, "end": 540}, ...]
    start/end are in milliseconds.
    """
    import json as _json
    try:
        logger.info("Starting AssemblyAI transcription...")
        # SpeechModel.best was introduced in newer SDK versions.
        # Fall back gracefully if the attribute doesn't exist.
        try:
            _speech_model = aai.SpeechModel.best
        except AttributeError:
            try:
                _speech_model = aai.SpeechModel.universal
            except AttributeError:
                _speech_model = None

        _cfg_kwargs = {"language_code": "hi"}
        if _speech_model is not None:
            _cfg_kwargs["speech_model"] = _speech_model

        config = aai.TranscriptionConfig(**_cfg_kwargs)
        transcriber = aai.Transcriber(config=config)
        transcript  = transcriber.transcribe(audio_path)
        if transcript.status == "error":
            raise RuntimeError(f"🔴Transcription failed: {transcript.error}")

        # ── SRT export ──
        try:
            srt_text = transcript.export_subtitles_srt()
        except AttributeError:
            logger.warning("Falling back to REST API subtitles endpoint.")
            url     = f"https://api.assemblyai.com/v2/transcripts/{transcript.id}/subtitles"
            headers = {"authorization": _AAI_KEY or getattr(aai.settings, "api_key", _AAI_KEY)}
            r       = requests.get(url, headers=headers, params={"subtitle_format": "srt"})
            r.raise_for_status()
            srt_text = r.text

        # ── Word-level timestamps ──
        words = []
        if hasattr(transcript, "words") and transcript.words:
            for w in transcript.words:
                words.append({
                    "text":  w.text,
                    "start": w.start,   # milliseconds
                    "end":   w.end,
                })
            logger.info(f"Got {len(words)} word timestamps from AssemblyAI.")
        else:
            logger.warning("No word timestamps available — karaoke will use even distribution.")

        logger.info("🟢AssemblyAI transcription completed successfully.")
        return srt_text, words
    except Exception as e:
        logger.exception("🔴AssemblyAI transcription failed.")
        raise


# =====================================================
# Caption routes (original — unchanged)
# =====================================================

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    uid               = uuid.uuid4().hex
    fname             = file.filename
    safe_fname        = fname.replace(" ", "_")
    video_path        = f"temp/temp_{uid}_{safe_fname}"
    audio_path        = f"temp/temp_{uid}_audio.wav"
    srt_path          = f"temp/captions_{uid}.srt"
    output_path       = f"videos/captioned_{uid}_{safe_fname}"
    output_video_name = f"captioned_{uid}_{safe_fname}"

    try:
        logger.info(f"Saving upload to {video_path}")
        content = await file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        logger.info("Extracting audio...")
        ffmpeg.input(video_path).output(
            audio_path, ac=1, ar=16000
        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        logger.info(f"Audio extracted: {audio_path}")
        loop               = asyncio.get_running_loop()
        transcribe_result  = await loop.run_in_executor(
            executor, transcribe_with_assemblyai, audio_path
        )
        import json as _json
        srt_text, words = transcribe_result
        save_srt_from_text(srt_text, srt_path)

        # Save word-level timestamps to JSON sidecar (for precise karaoke)
        words_path = srt_path.replace(".srt", "_words.json")
        with open(words_path, "w", encoding="utf-8") as wf:
            _json.dump(words, wf, ensure_ascii=False)
        logger.info(f"Words JSON saved: {words_path}")

        uploads[uid] = {
            "video_path":        video_path,
            "original_filename": fname,
            "srt_path":          srt_path,
            "words_path":        words_path,
            "output_name":       output_video_name,
            "output_path":       output_path,
        }
        return {
            "uid":               uid,
            "original_filename": fname,
            "srt_url":           f"/srt/{uid}",
            "video_url":         f"/original/{uid}",
        }
    except Exception as e:
        logger.exception("Upload failed")
        for p in [video_path, audio_path, srt_path, output_path]:
            try:
                if os.path.exists(p): os.remove(p)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/srt/{uid}", response_class=PlainTextResponse)
async def get_srt(uid: str):
    info = uploads.get(uid)
    if not info or not os.path.exists(info["srt_path"]):
        raise HTTPException(status_code=404, detail="SRT not found")
    with open(info["srt_path"], "r", encoding="utf-8") as f:
        return f.read()


@app.post("/srt/{uid}")
async def save_srt_endpoint(uid: str, payload: dict):
    info = uploads.get(uid)
    if not info:
        raise HTTPException(status_code=404, detail="UID not found")
    content = payload.get("content")
    if content is None:
        raise HTTPException(status_code=400, detail="No content provided")
    with open(info["srt_path"], "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"SRT updated for {uid}")
    return {"ok": True, "message": "SRT saved"}


class BurnRequest(BaseModel):
    font_size:       int   = 56
    text_color:      str   = "#FFFFFF"
    highlight_color: str   = "#FFFF00"  # word highlight colour as spoken
    outline_color:   str   = "#000000"
    bg_color:        str   = "#000000"
    bg_alpha:        int   = 100        # 0=opaque, 255=transparent
    position:        str   = "bottom_center"
    bold:            bool  = True


@app.post("/burn/{uid}")
async def burn_endpoint(uid: str, req: BurnRequest = BurnRequest()):
    import json as _json
    info = uploads.get(uid)
    if not info:
        raise HTTPException(status_code=404, detail="UID not found")

    # Load word timestamps if available
    words_json = None
    words_path = info.get("words_path")
    if words_path and os.path.exists(words_path):
        try:
            with open(words_path, "r", encoding="utf-8") as wf:
                words_json = _json.load(wf)
            logger.info(f"Loaded {len(words_json)} word timestamps for karaoke.")
        except Exception as e:
            logger.warning(f"Could not load words JSON: {e}")

    ok = burn_subtitles(
        info["video_path"], info["srt_path"], info["output_path"],
        font_size=req.font_size,
        text_color=req.text_color,
        highlight_color=req.highlight_color,
        outline_color=req.outline_color,
        bg_color=req.bg_color,
        bg_alpha=req.bg_alpha,
        position=req.position,
        bold=req.bold,
        words_json=words_json,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="FFmpeg failed")
    return {"ok": True, "video_url": f"/videos/{os.path.basename(info['output_path'])}"}


@app.get("/original/{uid}")
async def serve_original(uid: str):
    info = uploads.get(uid)
    if not info:
        raise HTTPException(status_code=404, detail="UID not found")
    return FileResponse(
        info["video_path"],
        media_type="video/mp4",
        filename=info["original_filename"]
    )


# =====================================================
# Storage / Cleanup helpers
# =====================================================

def _folder_stats(folder_path: str) -> dict:
    """Return file count and total size in bytes for a folder."""
    total_size  = 0
    total_files = 0
    files       = []
    if os.path.exists(folder_path):
        for fn in os.listdir(folder_path):
            fp = os.path.join(folder_path, fn)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                total_size  += size
                total_files += 1
                files.append({
                    "name":     fn,
                    "size":     size,
                    "modified": int(os.path.getmtime(fp)),
                })
    files.sort(key=lambda x: x["modified"], reverse=True)
    return {"count": total_files, "size_bytes": total_size, "files": files}


def _delete_folder_contents(folder_path: str) -> int:
    """Delete all files in a folder. Returns number of files deleted."""
    deleted = 0
    if os.path.exists(folder_path):
        for fn in os.listdir(folder_path):
            fp = os.path.join(folder_path, fn)
            try:
                if os.path.isfile(fp) or os.path.islink(fp):
                    os.unlink(fp)
                    deleted += 1
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
                    deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete {fp}: {e}")
    return deleted


# ── Storage stats endpoint ────────────────────────────────────

@app.get("/storage/stats")
async def storage_stats():
    """
    Return per-folder file counts, sizes, and file lists.
    Used by the UI cleanup panel to show what is on disk.
    """
    videos_stats = _folder_stats(VIDEOS_DIR)
    audio_stats  = _folder_stats(AUDIO_DIR)
    temp_stats   = _folder_stats(TEMP_DIR)
    total_bytes  = videos_stats["size_bytes"] + audio_stats["size_bytes"] + temp_stats["size_bytes"]
    return {
        "videos": videos_stats,
        "audio":  audio_stats,
        "temp":   temp_stats,
        "total_size_bytes": total_bytes,
    }


# ── Selective delete endpoints ────────────────────────────────

@app.delete("/storage/videos")
async def delete_videos():
    """Delete all files in the videos/ folder."""
    n = _delete_folder_contents(VIDEOS_DIR)
    # Also clear in-memory upload records whose output files are gone
    to_remove = [uid for uid, info in uploads.items()
                 if not os.path.exists(info.get("output_path", ""))]
    for uid in to_remove:
        del uploads[uid]
    logger.info(f"🗑 Deleted {n} video file(s).")
    return {"ok": True, "deleted": n, "message": f"Deleted {n} video file(s)."}


@app.delete("/storage/audio")
async def delete_audio():
    """Delete all files in the audio/ folder (generated TTS)."""
    n = _delete_folder_contents(AUDIO_DIR)
    logger.info(f"🗑 Deleted {n} audio file(s).")
    return {"ok": True, "deleted": n, "message": f"Deleted {n} audio file(s)."}


@app.delete("/storage/temp")
async def delete_temp():
    """Delete all files in the temp/ folder (uploads, SRT, WAV)."""
    n = _delete_folder_contents(TEMP_DIR)
    uploads.clear()
    logger.info(f"🗑 Deleted {n} temp file(s).")
    return {"ok": True, "deleted": n, "message": f"Deleted {n} temp file(s)."}


@app.delete("/storage/file")
async def delete_single_file(payload: dict):
    """
    Delete a single file by folder + filename.
    Body: { "folder": "videos"|"audio"|"temp", "filename": "xyz.mp4" }
    """
    folder_map = {"videos": VIDEOS_DIR, "audio": AUDIO_DIR, "temp": TEMP_DIR}
    folder  = payload.get("folder", "")
    fname   = payload.get("filename", "")
    if folder not in folder_map:
        raise HTTPException(status_code=400, detail=f"Invalid folder '{folder}'.")
    if not fname or ".." in fname or "/" in fname or "\\" in fname:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    fp = os.path.join(folder_map[folder], fname)
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="File not found.")
    os.unlink(fp)
    logger.info(f"🗑 Deleted single file: {fp}")
    return {"ok": True, "message": f"Deleted {fname}"}


# ── Full reset (keep for backward compat) ────────────────────

@app.delete("/reset")
async def reset_server():
    """Delete ALL generated files across videos, audio and temp folders."""
    total = 0
    for folder in [VIDEOS_DIR, TEMP_DIR, AUDIO_DIR]:
        total += _delete_folder_contents(folder)
    uploads.clear()
    logger.info(f"✅ Full reset — {total} files deleted.")
    return {"message": f"✅ All files deleted successfully. ({total} files removed)"}


@app.api_route("/", methods=["GET", "HEAD"])
async def serve_frontend(request: Request):
    # HEAD is sent by Render health checks, load balancers, and browsers.
    # Return 200 for HEAD without a body; serve index.html for GET.
    from fastapi.responses import Response as _Response
    index_path = os.path.join("frontend", "index.html")
    if request.method == "HEAD":
        return _Response(status_code=200)
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found."}


@app.get("/download/{filename}")
async def download_video(filename: str, background_tasks: BackgroundTasks):
    file_path = f"videos/{filename}"
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    background_tasks.add_task(os.remove, file_path)
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "ok"}


@app.get("/logdir")
def log_current_dir():
    cwd = '/opt/render/project/src/temp'
    try:
        for item in os.listdir(cwd):
            path = os.path.join(cwd, item)
            if os.path.isdir(path): logger.info(f"[DIR]  {path}")
            else: logger.info(f"[FILE] {path}")
    except Exception as e:
        logger.error(f"Error listing directory: {e}")


# =====================================================
# TTS routes — edge_tts (Microsoft Neural voices)
#              with gTTS fallback on 403 errors
# =====================================================

class TTSRequest(BaseModel):
    text:     str
    voice_id: str   = "hi_female"
    rate:     str   = "+0%"      # speed   e.g. "+20%", "-10%"
    volume:   str   = "+50%"     # volume  e.g. "+0%", "+100%"
    pitch:    str   = "+0Hz"     # pitch   e.g. "+5Hz", "-5Hz"
    # ── Echo settings ──────────────────────────────
    echo_enabled:   bool  = False
    echo_delay:     int   = 500    # ms between echo repeats (50–2000)
    echo_decay:     float = 0.4    # echo volume falloff per repeat (0.1–0.9)
    echo_taps:      int   = 1      # number of echo repeats (1–4)
    # ── Reverb settings ────────────────────────────
    reverb_enabled: bool  = False
    reverb_amount:  float = 0.3    # room size/wetness (0.1–1.0)


@app.get("/tts/voices")
async def list_voices():
    """Return all available edge_tts voices."""
    return {"voices": TTS_VOICES}


@app.post("/tts/synthesize")
async def tts_synthesize(req: TTSRequest):
    """
    Generate MP3 using edge_tts (Microsoft Neural voices).
    Automatically retries 3× on 403 errors, then falls back to gTTS.
    Returns { ok, audio_url, filename }.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(req.text) > 5000:
        raise HTTPException(status_code=400, detail="Text exceeds 5000-character limit.")
    if req.voice_id not in VOICE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice_id '{req.voice_id}'. Available: {list(VOICE_MAP.keys())}"
        )

    # Use voice preset defaults if user didn't override
    voice  = VOICE_MAP[req.voice_id]
    rate   = req.rate   if req.rate   != "+0%"   else voice["rate"]
    volume = req.volume if req.volume != "+50%"  else voice["volume"]
    pitch  = req.pitch  if req.pitch  != "+0Hz"  else voice["pitch"]

    try:
        loop     = asyncio.get_running_loop()
        mp3_path = await loop.run_in_executor(
            executor,
            synthesize_edge_tts,
            req.text, req.voice_id, rate, volume, pitch,
        )

        # ── Apply echo / reverb if requested ──
        any_effect = req.echo_enabled or req.reverb_enabled
        if any_effect:
            uid_fx    = uuid.uuid4().hex
            fx_out    = os.path.join(AUDIO_DIR, f"tts_fx_{uid_fx}.mp3")
            mp3_path  = await loop.run_in_executor(
                executor,
                apply_audio_effects,
                mp3_path, fx_out,
                req.echo_enabled,  req.echo_delay,  req.echo_decay, req.echo_taps,
                req.reverb_enabled, req.reverb_amount,
            )

        filename = os.path.basename(mp3_path)
        return {"ok": True, "audio_url": f"/audio/{filename}", "filename": filename}

    except Exception as e:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts/download/{filename}")
async def download_tts(filename: str, background_tasks: BackgroundTasks):
    """Serve the MP3 and delete it afterwards."""
    fp = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="Audio file not found")
    background_tasks.add_task(os.remove, fp)
    return FileResponse(fp, media_type="audio/mpeg", filename=filename)


# -----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    _port = int(os.getenv("PORT", 8000))
    logger.info(f"Server starting on http://0.0.0.0:{_port}")
    uvicorn.run(app, host="0.0.0.0", port=_port, reload=True)