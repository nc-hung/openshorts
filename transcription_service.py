import os
import json
import math
import tempfile
import subprocess
from typing import Optional, List, Dict, Any


# BCP-47 language tags for Google Cloud STT.
_LANG_TAGS = {
    "vi": "vi-VN",
    "en": "en-US",
    "zh": "zh-CN",
    "pt": "pt-BR",
    "es": "es-ES",
    "ja": "ja-JP",
    "ko": "ko-KR",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "ru": "ru-RU",
    "ar": "ar-SA",
    "id": "id-ID",
    "th": "th-TH",
    "ms": "ms-MY",
    "fil": "fil-PH",
}

# Google sync Recognize: max ~1 min / ~10 MB.  We use 55 s chunks for safety.
_GOOGLE_CHUNK_SEC = 55.0


class TranscriptionService:
    """
    Unified transcription service for OpenShorts.

    Provider logic
    --------------
    * ``auto`` (default)    → local faster-whisper;  fallback Google STT if quality poor.
    * ``local``             → local faster-whisper only.
    * ``google``            → Google Cloud STT *first*;  raises if not configured.

    Language
    --------
    Pass ``"auto"`` or ``None`` to let faster-whisper auto-detect.
    """

    def __init__(self):
        self._local_model = None
        self._cfg = self._load_config()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _load_config(self) -> dict:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        # Try to derive project_id from the service-account JSON when not set.
        if not project:
            creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            if creds_path and os.path.exists(creds_path):
                try:
                    with open(creds_path) as f:
                        data = json.load(f)
                    project = data.get("project_id", "")
                except Exception:
                    pass

        return {
            "provider": os.environ.get("ASR_PROVIDER", "auto"),
            "language": os.environ.get("ASR_LANGUAGE", "vi"),
            "local_model": os.environ.get("ASR_LOCAL_MODEL", "large-v3"),
            "hotwords": os.environ.get("ASR_HOTWORDS", ""),
            "google_creds": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
            "google_project": project,
            "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
        }

    # ------------------------------------------------------------------
    # Audio normalisation  (mono 16-bit 16 kHz WAV)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_audio(video_path: str) -> str:
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1",
                wav,
            ],
            check=True, capture_output=True,
        )
        return wav

    @staticmethod
    def _wav_duration(wav_path: str) -> float:
        """Return duration in seconds of a WAV file via ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            wav_path,
        ]
        out = subprocess.check_output(cmd).decode().strip()
        return float(out)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def transcribe(
        self,
        video_path: str,
        language: Optional[str] = None,
        provider: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe video audio → text + word-level timestamps.

        Parameters
        ----------
        language : str or None
            Language code (``"vi"``, ``"en"``, …).  Pass ``"auto"`` or ``None``
            to let faster-whisper auto-detect.
        provider : str or None
            ``"auto"``, ``"local"``, or ``"google"``.
        """
        raw_lang = language or self._cfg["language"]
        # "auto" from the UI → let whisper auto-detect
        whisper_lang = None if raw_lang in (None, "", "auto") else raw_lang
        prov = provider or self._cfg["provider"]
        hw = hotwords or self._cfg["hotwords"]

        wav_path = self._normalize_audio(video_path)
        try:
            # ---- explicit google ----
            if prov == "google":
                result = self._transcribe_google(wav_path, raw_lang, hw)
                result["provider"] = "google"
                result["quality_flags"] = self._quality_flags(result)
                result.setdefault("corrected_words", [])
                return result

            # ---- local ----
            result = self._transcribe_local(wav_path, whisper_lang, hw)
            result["provider"] = "local"

            flags = self._quality_flags(result)
            result["quality_flags"] = flags

            # ---- fallback to Google STT (auto mode + poor quality) ----
            if flags and prov == "auto" and self._has_google():
                try:
                    g = self._transcribe_google(wav_path, raw_lang, hw)
                    g["provider"] = "google"
                    g["quality_flags"] = self._quality_flags(g)
                    result = g
                except Exception as exc:
                    print(f"[transcription] Google STT fallback failed: {exc}")

            # ---- Gemini word-level correction (local results only) ----
            if result["provider"] == "local" and (
                flags or self._has_single_char_tokens(result)
            ) and self._cfg["gemini_api_key"]:
                result = self._correct_with_gemini(result, raw_lang)
                result["provider"] = "local+gemini"

            result.setdefault("corrected_words", [])
            return result

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    # ------------------------------------------------------------------
    # Local  (faster-whisper)
    # ------------------------------------------------------------------
    def _transcribe_local(self, audio_path: str, language: Optional[str],
                          hotwords: str) -> dict:
        from faster_whisper import WhisperModel

        if self._local_model is None:
            print(f"  Downloading / loading faster-whisper model: {self._cfg['local_model']}  (this may take several minutes on first run)")
            import sys as _sys
            _sys.stdout.flush()
            self._local_model = WhisperModel(
                self._cfg["local_model"], device="cpu", compute_type="int8",
            )
            print(f"  ASR model ready: {self._cfg['local_model']}")

        hw_list = [w.strip() for w in hotwords.split(",") if w.strip()]
        hw_str = ",".join(hw_list) or None

        segments, info = self._local_model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            hotwords=hw_str,
        )

        segs = []
        full_text = ""
        for seg in segments:
            d = {"text": seg.text, "start": seg.start, "end": seg.end, "words": []}
            if seg.words:
                for w in seg.words:
                    d["words"].append({
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                        "probability": getattr(w, "probability", 1.0),
                    })
            segs.append(d)
            full_text += (seg.text or "") + " "

        return {
            "text": full_text.strip(),
            "segments": segs,
            "language": info.language,
            "language_probability": getattr(info, "language_probability", 1.0),
        }

    # ------------------------------------------------------------------
    # Quality heuristics  (Vietnamese-aware)
    # ------------------------------------------------------------------
    @staticmethod
    def _quality_flags(result: dict) -> List[str]:
        flags = []
        if result.get("language_probability", 1.0) < 0.5:
            flags.append("low_language_confidence")

        total = 0
        low_conf = 0
        single_char = 0
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                total += 1
                if w.get("probability", 1.0) < 0.5:
                    low_conf += 1
                wt = w.get("word", "").strip().lower()
                if len(wt) <= 1 and wt.isalpha():
                    single_char += 1

        if total and low_conf / total > 0.2:
            flags.append("too_many_low_confidence_words")
        if single_char > 3:
            flags.append("numerous_gibberish_tokens")
        return flags

    @staticmethod
    def _has_single_char_tokens(result: dict) -> bool:
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                wt = w.get("word", "").strip().lower()
                if len(wt) <= 1 and wt.isalpha():
                    return True
        return False

    # ------------------------------------------------------------------
    # Google Cloud Speech-to-Text V2  (chirp_3)
    # ------------------------------------------------------------------
    def _has_google(self) -> bool:
        return bool(self._cfg["google_project"])

    @staticmethod
    def _google_lang_tag(language: str) -> str:
        """Return a BCP-47 tag (e.g. ``"vi-VN"``) for the given short code.
        ``"auto"`` → ``"vi-VN"``  (Google needs an explicit code)."""
        tag = _LANG_TAGS.get(language)
        if tag:
            return tag
        if len(language) == 2:
            return f"{language}-{language.upper()}"
        return "vi-VN"  # default

    def _transcribe_google(self, audio_path: str, language: str,
                           hotwords: str) -> dict:
        try:
            from google.cloud.speech_v2 import SpeechClient
            from google.cloud.speech_v2.types import cloud_speech as cs
        except ImportError:
            raise ImportError(
                "google-cloud-speech is not installed. "
                "Run: pip install google-cloud-speech"
            )

        if not self._has_google():
            raise RuntimeError(
                "Google Cloud STT selected but GOOGLE_CLOUD_PROJECT is not set "
                "and could not be derived from GOOGLE_APPLICATION_CREDENTIALS."
            )

        client = SpeechClient()
        lang_tag = self._google_lang_tag(language)

        recognizer = (
            f"projects/{self._cfg['google_project']}/locations/global/recognizers/_"
        )

        # Build hotword phrases for SpeechAdaptation (v2 API shape).
        hw_list = [w.strip() for w in hotwords.split(",") if w.strip()]
        adaptation = None
        if hw_list:
            adaptation = cs.SpeechAdaptation(
                phrase_sets=[
                    cs.AdaptationPhraseSet(
                        inline_phrase_set=cs.PhraseSet(
                            phrases=[
                                cs.PhraseSet.Phrase(value=hw, boost=15)
                                for hw in hw_list
                            ]
                        )
                    )
                ]
            )

        duration = self._wav_duration(audio_path)
        num_chunks = max(1, math.ceil(duration / _GOOGLE_CHUNK_SEC))

        if num_chunks == 1:
            # Short enough for a single sync Recognize call.
            return self._google_recognize(
                client, audio_path, recognizer, lang_tag, adaptation,
            )

        # Long audio: split into chunks, transcribe each, merge.
        print(f"  Google STT: splitting {duration:.0f}s audio into {num_chunks} chunks")
        chunk_dir = tempfile.mkdtemp(prefix="gstt_")
        try:
            all_segments = []
            for i in range(num_chunks):
                start_s = i * _GOOGLE_CHUNK_SEC
                chunk_path = os.path.join(chunk_dir, f"chunk_{i:04d}.wav")
                subprocess.run(
                    ["ffmpeg", "-y",
                     "-ss", str(start_s),
                     "-t", str(_GOOGLE_CHUNK_SEC),
                     "-i", audio_path,
                     "-c", "copy", chunk_path],
                    check=True, capture_output=True,
                )

                chunk_result = self._google_recognize(
                    client, chunk_path, recognizer, lang_tag, adaptation,
                )
                # Offset all timestamps by chunk start.
                for seg in chunk_result["segments"]:
                    seg["start"] += start_s
                    seg["end"] += start_s
                    for w in seg.get("words", []):
                        w["start"] += start_s
                        w["end"] += start_s
                all_segments.extend(chunk_result["segments"])
                os.remove(chunk_path)

            merged_text = " ".join(
                s["text"] for s in all_segments
            )
            return {
                "text": merged_text,
                "segments": all_segments,
                "language": language,
                "language_probability": 1.0,
            }
        finally:
            for f in os.listdir(chunk_dir):
                os.remove(os.path.join(chunk_dir, f))
            os.rmdir(chunk_dir)

    def _google_recognize(self, client, audio_path, recognizer, lang_tag,
                          adaptation) -> dict:
        """Single sync Recognize call (audio must be ≤ ~60 s)."""
        from google.cloud.speech_v2.types import cloud_speech as cs

        with open(audio_path, "rb") as f:
            content = f.read()

        config = cs.RecognitionConfig(
            auto_decoding_config=cs.AutoDetectDecodingConfig(),
            model="chirp_3",
            language_codes=[lang_tag],
            adaptation=adaptation,
            features=cs.RecognitionFeatures(
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            ),
        )

        resp = client.recognize(
            request=cs.RecognizeRequest(
                recognizer=recognizer,
                config=config,
                content=content,
            )
        )

        segs = []
        full_text = ""
        for r in resp.results:
            for alt in r.alternatives:
                seg = {"text": alt.transcript, "start": 0.0, "end": 0.0, "words": []}
                for wd in alt.words:
                    seg["words"].append({
                        "word": wd.word,
                        "start": wd.start_offset.total_seconds(),
                        "end": wd.end_offset.total_seconds(),
                        "probability": 1.0,
                    })
                if seg["words"]:
                    seg["start"] = seg["words"][0]["start"]
                    seg["end"] = seg["words"][-1]["end"]
                segs.append(seg)
                full_text += alt.transcript + " "

        return {
            "text": full_text.strip(),
            "segments": segs,
            "language": lang_tag,
            "language_probability": 1.0,
        }

    # ------------------------------------------------------------------
    # Gemini word-level correction
    # ------------------------------------------------------------------
    def _correct_with_gemini(self, result: dict, language: str) -> dict:
        from google import genai

        corrections = []
        for si, seg in enumerate(result.get("segments", [])):
            words = seg.get("words", [])
            for wi, w in enumerate(words):
                prob = w.get("probability", 1.0)
                wt = w.get("word", "").strip().lower()
                is_gibberish = len(wt) <= 1 and wt.isalpha()
                if prob >= 0.5 and not is_gibberish:
                    continue
                ctx_start = max(0, wi - 3)
                ctx_end = min(len(words), wi + 4)
                before = " ".join(words[j]["word"] for j in range(ctx_start, wi))
                after = " ".join(words[j]["word"] for j in range(wi + 1, ctx_end))
                corrections.append({
                    "seg_idx": si,
                    "word_idx": wi,
                    "original": w["word"],
                    "context_before": before,
                    "context_after": after,
                })

        if not corrections:
            return result

        client = genai.Client(api_key=self._cfg["gemini_api_key"])

        prompt = (
            f"You are a {language or 'auto'} speech-to-text post-processor. "
            f"For each word, decide if it was misheard given its context. "
            f"Return ONLY a JSON array: [{{\"seg_idx\":0,\"word_idx\":5,\"corrected\":\"ích\"}}] "
            f"Omit words that are already correct.\n\n"
            f"Corrections needed:\n{json.dumps(corrections, ensure_ascii=False, indent=2)}"
        )

        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt,
            )
            text = resp.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            gem = json.loads(text)
            fixed = []
            for c in gem:
                si, wi = c["seg_idx"], c["word_idx"]
                segs = result.get("segments", [])
                if si < len(segs) and wi < len(segs[si].get("words", [])):
                    orig = segs[si]["words"][wi]["word"]
                    segs[si]["words"][wi]["word"] = c["corrected"]
                    fixed.append({
                        "original": orig, "corrected": c["corrected"],
                        "seg_idx": si, "word_idx": wi,
                    })

            result["text"] = " ".join(
                w["word"] for seg in result["segments"] for w in seg["words"]
            )
            result["corrected_words"] = fixed
        except Exception as exc:
            print(f"  Gemini correction failed: {exc}")

        return result
