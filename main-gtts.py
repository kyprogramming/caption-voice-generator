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
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
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
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
logger.info("Using AssemblyAI for transcription (Hindi mode).")

# =====================================================
# gTTS Voice catalogue
# =====================================================
#
# gTTS uses Google Text-to-Speech API (free, needs internet).
# It does not natively support male/female selection —
# we simulate it using speed/pitch post-processing via ffmpeg:
#   Female : normal speed, normal pitch
#   Male   : slightly slower + pitch shifted down
#
# Supported languages: English (en) and Hindi (hi)
# =====================================================

TTS_VOICES = [
    {
        "voice_id": "en_female",
        "lang":     "en",
        "gender":   "Female",
        "name":     "English Female",
        "label":    "English Female (Google TTS)",
        "gtts_lang": "en",
        "slow":     False,
        "pitch_shift": 0.0,    # semitones
        "speed_factor": 1.0,
    },
    {
        "voice_id": "en_male",
        "lang":     "en",
        "gender":   "Male",
        "name":     "English Male",
        "label":    "English Male (Google TTS, deep tone)",
        "gtts_lang": "en",
        "slow":     False,
        "pitch_shift": -3.0,   # shift down 3 semitones for male tone
        "speed_factor": 0.95,
    },
    {
        "voice_id": "hi_female",
        "lang":     "hi",
        "gender":   "Female",
        "name":     "Hindi Female",
        "label":    "Hindi Female (Google TTS)",
        "gtts_lang": "hi",
        "slow":     False,
        "pitch_shift": 0.0,
        "speed_factor": 1.0,
    },
    {
        "voice_id": "hi_male",
        "lang":     "hi",
        "gender":   "Male",
        "name":     "Hindi Male",
        "label":    "Hindi Male (Google TTS, deep tone)",
        "gtts_lang": "hi",
        "slow":     False,
        "pitch_shift": -3.0,   # shift down 3 semitones for male tone
        "speed_factor": 0.95,
    },
]

# Quick lookup dict
VOICE_MAP = {v["voice_id"]: v for v in TTS_VOICES}


# =====================================================
# gTTS synthesis function
# =====================================================

def synthesize_gtts(text: str, voice_id: str, speed: float, pitch: float) -> str:
    """
    Synthesize speech using gTTS (Google Text-to-Speech, free).
    Steps:
      1. Generate raw MP3 via gTTS
      2. Apply voice pitch preset (male = -3 semitones)
      3. Apply user speed + pitch adjustments via ffmpeg
      4. Encode final 192 kbps MP3
    Returns absolute path to final MP3 in AUDIO_DIR.
    """
    from gtts import gTTS

    if voice_id not in VOICE_MAP:
        raise ValueError(f"Unknown voice_id '{voice_id}'.")

    meta      = VOICE_MAP[voice_id]
    uid       = uuid.uuid4().hex
    raw_mp3   = os.path.join(TEMP_DIR,  f"tts_raw_{uid}.mp3")
    mp3_out   = os.path.join(AUDIO_DIR, f"tts_{uid}.mp3")

    logger.info(f"gTTS: voice={voice_id} | speed={speed} | pitch={pitch} | chars={len(text)}")

    # Step 1 — gTTS → raw MP3
    tts = gTTS(text=text, lang=meta["gtts_lang"], slow=meta["slow"])
    tts.save(raw_mp3)
    logger.info(f"gTTS raw MP3: {raw_mp3}")

    # Step 2 & 3 — combine voice preset + user adjustments
    # Total pitch = voice preset pitch_shift + user pitch
    total_pitch = meta["pitch_shift"] + float(pitch)
    # Total speed = voice preset speed_factor * user speed
    total_speed = meta["speed_factor"] * max(0.5, min(2.0, float(speed)))
    total_speed = max(0.5, min(2.0, total_speed))

    filters = []

    if total_speed != 1.0:
        # atempo range is 0.5–2.0; chain two filters if needed
        if total_speed < 0.5:
            filters.append("atempo=0.5")
            filters.append(f"atempo={total_speed/0.5:.3f}")
        elif total_speed > 2.0:
            filters.append("atempo=2.0")
            filters.append(f"atempo={total_speed/2.0:.3f}")
        else:
            filters.append(f"atempo={total_speed:.3f}")

    if total_pitch != 0.0:
        base_rate      = 24000          # gTTS outputs 24 kHz
        semitone_ratio = 2 ** (total_pitch / 12.0)
        shifted_rate   = int(base_rate * semitone_ratio)
        filters.append(f"asetrate={shifted_rate}")
        filters.append(f"aresample={base_rate}")

    af_str = ",".join(filters) if filters else "anull"

    try:
        (
            ffmpeg
            .input(raw_mp3)
            .output(
                mp3_out,
                af=af_str,
                audio_bitrate="192k",
                acodec="libmp3lame"
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        logger.info(f"MP3 saved: {mp3_out}")
    except ffmpeg.Error as e:
        stderr_msg = e.stderr.decode("utf-8", errors="replace") if e.stderr else "no stderr"
        logger.error(f"ffmpeg TTS post-process failed:\n{stderr_msg}")
        raise RuntimeError(f"ffmpeg error: {stderr_msg}") from e
    finally:
        try:
            os.remove(raw_mp3)
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


def convert_srt_to_ass_karaoke(srt_file, ass_file):
    try:
        logger.info("Starting conversion: SRT -> ASS")
        srt_content = ""
        with open(srt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                srt_content += line
        srt_content = srt_content.strip()

        mukta_path = os.path.join(FONTS_DIR, "NotoSansDevanagari-Regular.ttf")
        fontname   = "NotoSansDevanagari" if os.path.exists(mukta_path) else "Mukta"
        logger.info(f"Using font: {fontname}")

        ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes
Collisions: Normal
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},56,&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,3,2,5,30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        srt_content = srt_content.replace("\r\n", "\n").replace("\r", "\n")
        blocks      = srt_content.split("\n\n")
        ass_lines   = []

        for i, block in enumerate(blocks):
            lines = block.split("\n")
            if len(lines) < 3:
                continue
            time_line = lines[1].strip()
            text      = ' '.join(lines[2:]).strip()
            if not text:
                continue
            start, end = time_line.split(" --> ")

            def srt_to_ass_time(t):
                t, ms = t.split(",")
                h, m, s = t.split(":")
                return f"{h}:{m}:{s}.{ms[:2]}"

            ass_start = srt_to_ass_time(start)
            ass_end   = srt_to_ass_time(end)
            words     = text.split()
            dpw       = 3.0 / max(len(words), 1)
            kar       = ''.join([f"{{\\k{int(dpw * 100)}}}{w} " for w in words])
            ass_lines.append(
                f"Dialogue: 0,{ass_start},{ass_end},Default,,0,0,0,,{{\\an5}}{kar.strip()}"
            )

        with open(ass_file, 'w', encoding='utf-8-sig') as f:
            f.write(ass_content + "\n" + "\n".join(ass_lines))

        logger.info(f"🟢Conversion successful: {ass_file}")
        return True

    except Exception as e:
        logger.exception("🔴ASS conversion failed")
        return False


def _safe_ass_path(p: str) -> str:
    """
    Convert path to ffmpeg-safe format on Windows.
    Backslashes → forward-slashes, drive colon escaped (C: → C\\:).
    Linux/Mac paths returned unchanged.
    """
    p = p.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\:" + p[2:]
    return p


def burn_subtitles(video_path, srt_file, output_path):
    logger.info("🔥Burning subtitles to video...")
    ass_file = srt_file.replace('.srt', '.ass')
    if not convert_srt_to_ass_karaoke(srt_file, ass_file):
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
# AssemblyAI transcription (original — unchanged)
# -----------------------------------------------------
def transcribe_with_assemblyai(audio_path: str) -> str:
    try:
        logger.info("Starting AssemblyAI transcription...")
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            language_code="hi"
        )
        transcriber = aai.Transcriber(config=config)
        transcript  = transcriber.transcribe(audio_path)
        if transcript.status == "error":
            raise RuntimeError(f"🔴Transcription failed: {transcript.error}")
        try:
            srt_text = transcript.export_subtitles_srt()
        except AttributeError:
            logger.warning("Falling back to REST API subtitles endpoint.")
            url     = f"https://api.assemblyai.com/v2/transcripts/{transcript.id}/subtitles"
            headers = {"authorization": aai.settings.api_key}
            r       = requests.get(url, headers=headers, params={"subtitle_format": "srt"})
            r.raise_for_status()
            srt_text = r.text
        logger.info("🟢AssemblyAI transcription completed successfully.")
        return srt_text
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

        loop     = asyncio.get_running_loop()
        srt_text = await loop.run_in_executor(
            executor, transcribe_with_assemblyai, audio_path
        )
        save_srt_from_text(srt_text, srt_path)

        uploads[uid] = {
            "video_path":        video_path,
            "original_filename": fname,
            "srt_path":          srt_path,
            "output_name":       output_video_name,
            "output_path":       output_path,
        }
        logger.info(f"Upload complete: uid={uid}")
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
                if os.path.exists(p):
                    os.remove(p)
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
    srt_path = info.get("srt_path")
    if not srt_path:
        raise HTTPException(status_code=404, detail="No SRT path")
    content = payload.get("content")
    if content is None:
        raise HTTPException(status_code=400, detail="No content provided")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"SRT updated for {uid}")
    return {"ok": True, "message": "SRT saved"}


@app.post("/burn/{uid}")
async def burn_endpoint(uid: str):
    info = uploads.get(uid)
    if not info:
        raise HTTPException(status_code=404, detail="UID not found")
    ok = burn_subtitles(info["video_path"], info["srt_path"], info["output_path"])
    if not ok:
        raise HTTPException(status_code=500, detail="FFmpeg failed")
    video_file = os.path.basename(info["output_path"])
    return {"ok": True, "video_url": f"/videos/{video_file}"}


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


@app.delete("/reset")
async def reset_server():
    for folder in ["videos", "temp", "audio"]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                fp = os.path.join(folder, f)
                try:
                    if os.path.isfile(fp) or os.path.islink(fp):
                        os.unlink(fp)
                    elif os.path.isdir(fp):
                        shutil.rmtree(fp)
                except Exception as e:
                    logger.error(f"Failed to delete {fp}: {e}")
    uploads.clear()
    logger.info("✅ All files deleted successfully.")
    return {"message": "✅ All files deleted successfully."}


@app.get("/")
async def serve_frontend():
    index_path = os.path.join("frontend", "index.html")
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


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/logdir")
def log_current_dir():
    cwd = '/opt/render/project/src/temp'
    try:
        for item in os.listdir(cwd):
            path = os.path.join(cwd, item)
            if os.path.isdir(path):
                logger.info(f"[DIR]  {path}")
            else:
                logger.info(f"[FILE] {path}")
    except Exception as e:
        logger.error(f"Error listing directory: {e}")


# =====================================================
# TTS routes — gTTS (Google TTS, free, all languages)
# =====================================================

class TTSRequest(BaseModel):
    text:     str
    voice_id: str   = "en_female"   # en_female | en_male | hi_female | hi_male
    speed:    float = 1.0            # 0.5 – 2.0  (user adjustment)
    pitch:    float = 0.0            # semitones   (user adjustment, -12 to +12)


@app.get("/tts/voices")
async def list_voices():
    """Return all available gTTS voices."""
    return {"voices": TTS_VOICES}


@app.post("/tts/synthesize")
async def tts_synthesize(req: TTSRequest):
    """
    Generate MP3 using gTTS (Google Text-to-Speech, free).
    Supports English + Hindi, male + female (pitch-shifted).
    Requires internet connection.
    Returns { ok, audio_url, filename }.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(req.text) > 3000:
        raise HTTPException(status_code=400, detail="Text exceeds 3000-character limit.")
    if req.voice_id not in VOICE_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice_id '{req.voice_id}'. "
                   f"Available: {list(VOICE_MAP.keys())}"
        )

    try:
        loop     = asyncio.get_running_loop()
        mp3_path = await loop.run_in_executor(
            executor,
            synthesize_gtts,
            req.text, req.voice_id, req.speed, req.pitch,
        )
        filename = os.path.basename(mp3_path)
        return {"ok": True, "audio_url": f"/audio/{filename}", "filename": filename}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts/download/{filename}")
async def download_tts(filename: str, background_tasks: BackgroundTasks):
    """Serve the generated MP3 and delete it from disk afterwards."""
    fp = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="Audio file not found")
    background_tasks.add_task(os.remove, fp)
    return FileResponse(fp, media_type="audio/mpeg", filename=filename)


# -----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Server starting on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)