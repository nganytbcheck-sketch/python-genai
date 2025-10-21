# pip install google-genai
import os, mimetypes, struct
from google import genai
from google.genai import types

TEXT = """Điều đáng sợ nhất trong tình yêu… không phải là người khác làm bạn tổn thương, mà là khi chính bạn cứ chọn người khiến mình tổn thương – hết lần này đến lần khác.
Bạn nói đó là duyên, là số phận, nhưng tâm lý học gọi đó là vết thương chưa khép.
Thực ra, bạn không chọn họ bằng lý trí… bạn chọn họ bằng nỗi sợ bị bỏ rơi, bằng ký ức chưa lành, bằng những mảnh cảm xúc bạn tưởng đã quên.
Và rồi, bạn gọi đó là tình yêu."""

def save_bin(path, data):
    with open(path, "wb") as f:
        f.write(data)
    print("Saved:", path)

def parse_audio_mime_type(mime_type: str):
    bits_per_sample, rate = 16, 24000
    for p in [x.strip() for x in mime_type.split(";")]:
        if p.lower().startswith("rate="):
            try: rate = int(p.split("=",1)[1])
            except: pass
        if "audio/L" in p:
            try: bits_per_sample = int(p.split("L",1)[1])
            except: pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    params = parse_audio_mime_type(mime_type)
    bps, sr, ch = params["bits_per_sample"], params["rate"], 1
    data_size = len(audio_data)
    block_align = ch * (bps // 8)
    byte_rate = sr * block_align
    import struct
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ", 16,
        1, ch, sr, byte_rate, block_align, bps, b"data", data_size
    )
    return header + audio_data

def tts_generate():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = "gemini-2.5-pro-preview-tts"

    cfg = types.GenerateContentConfig(
        temperature=0.3,
        seed=42,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Zephyr"   # hoặc "vi-VN-Wavenet-D" nếu có
                )
            ),
            speaking_rate=0.90,   # podcast, chậm vừa
            pitch=-1.0,           # trầm ấm
        ),
    )

    stream = client.models.generate_content_stream(
        model=model,
        contents=types.UserContent(parts=[types.Part.from_text(TEXT)]),
        config=cfg,
    )

    i = 0
    for chunk in stream:
        cands = getattr(chunk, "candidates", None)
        if not cands: continue
        parts = getattr(cands[0].content, "parts", None)
        if not parts: continue
        p0 = parts[0]
        if getattr(p0, "inline_data", None) and p0.inline_data.data:
            mime = p0.inline_data.mime_type
            data = p0.inline_data.data
            ext = mimetypes.guess_extension(mime) or ".wav"
            if ext == ".wav" and not mime.endswith("wav"):
                data = convert_to_wav(data, mime)
            save_bin(f"tamlyhocgenz_voice_{i}{ext}", data)
            i += 1

if __name__ == "__main__":
    tts_generate()
