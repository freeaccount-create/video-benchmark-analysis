# -*- coding: utf-8 -*-


import argparse
import base64
import datetime
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from moviepy import VideoFileClip, concatenate_videoclips  # type: ignore
except ImportError:  # pragma: no cover
    VideoFileClip = None
    concatenate_videoclips = None

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None


LOGGER = logging.getLogger(__name__)


API_VERSION = "v2.03"


# DEFAULT_CONTINUITY_PROMPT = (
#     "Maintain a consistent visual style and worldview. Keep scenes and lighting stable, and ensure character clothing, hairstyles, body types, and expressions are consistent, adjusting movements only according to the plot. If reference images are used, strictly adhere to the character designs; character positions must not change, and shot transitions must be smooth and natural. Narration is not required; it serves only as plot prompts. The final frame of the generated video must show all characters' frontal views and current positions. Mandatory: If my script is in Chinese, speak Chinese; if in English, speak English, maintaining language consistency. Do not use subtitles."
# )
DEFAULT_CONTINUITY_PROMPT = (
    "Maintain a consistent visual style and worldview. Keep scenes and lighting stable, and ensure character clothing, hairstyles, body types, and expressions are consistent, adjusting movements only according to the plot. If reference images are used, strictly adhere to the character designs; character positions must not change, and shot transitions must be smooth and natural. Narration is not required; it serves only as plot prompts. The final frame of the generated video must show all characters' frontal views and current positions. Mandatory: All character dialogue must be in English only. Do not use subtitles."
)

CONTINUITY_PROMPT = DEFAULT_CONTINUITY_PROMPT

REFERENCE_MODE_CHOICES: Sequence[str] = ("first", "style", "asset")

# ✅ List of keywords for content-sensitive errors (for quickly determining if a node needs to be skipped)
CONTENT_MODERATION_KEYWORDS: Sequence[str] = (
    "moderation", "authentication", "content_sensitive", "violation", "sensitive", 
    "policy", "refused", "rejected", "inappropriate", "blocked",
    "review", "prohibited", "not_allowed", "unsafe", "violation"
)


STYLE_PROMPTS: Dict[str, str] = {
    "anime": (
        "Overall visual requirements: High-quality 2D anime rendering style, hand-drawn anime characters, skin tones and textures with anime feel, and fictional anime scenes as backgrounds; realistic/live-action or real-life photography elements are prohibited."
    ),
    "realistic": (
        "Overall visual requirements: High-quality photorealistic style with rich lighting details on characters and environments, materials and textures close to the real world. Prohibit cartoon or exaggerated strokes. Ensure colors and lighting follow real-world physics. Avoid real celebrities or any elements that may be prohibited by the API as reference images."
    ),
    "animated": (
        "Overall visual requirements: Animation/cartoon style, supporting 2D or 3D rendering, clear character lines and outlines, saturated and layered colors. Moderate exaggeration of actions and expressions is allowed, but maintain overall tone consistency."
    ),
    "painterly": (
        "Overall visual requirements: Artistic painting style, presenting thick brushstrokes or watercolor blending textures. Artistic textures and brush marks are allowed. Overall colors and composition should be unified to create an artistic atmosphere."
    ),
    "abstract": (
        "Overall visual requirements: Abstract/experimental style, encouraging surreal, glitch art, or non-traditional composition techniques. Breaking realistic rules is allowed to highlight visual impact and creative expression, but the core subject must remain recognizable."
    ),
}


STYLE_ORDER: Sequence[str] = tuple(STYLE_PROMPTS.keys())
STYLE_CHOICES: Sequence[str] = tuple(list(STYLE_ORDER) + ["all"])
DEFAULT_STYLE_KEY = "anime"
STYLE_ALIASES: Dict[str, str] = {
    **{key: key for key in STYLE_PROMPTS},
    "anime": "anime",
    "cartoon": "anime",
    "animation": "anime",
    "realistic": "realistic",
    "photorealistic": "realistic",
    "true": "realistic",
    "animated": "animated",
    "cartoon style": "animated",
    "animation style": "animated",
    "stylized": "animated",
    "painterly": "painterly",
    "artistic": "painterly",
    "painting": "painterly",
    "abstract": "abstract",
    "experimental": "abstract",
    "glitch": "abstract",
    "surreal": "abstract",
    "all": "all",
    "all styles": "all",
}


def resolve_style_key(value: Optional[str]) -> str:
    if not value:
        return DEFAULT_STYLE_KEY

    normalized = value.strip().lower()
    resolved = STYLE_ALIASES.get(normalized)
    if resolved:
        return resolved

    resolved = STYLE_ALIASES.get(value.strip())
    if resolved:
        return resolved

    raise ValueError(
        f"Unknown style: {value}, options include: {', '.join(sorted(STYLE_CHOICES))}"
    )


SORA_SUPPORTED_SIZES: Sequence[str] = ("1280x720", "720x1280", "1024x1792", "1792x1024")
SORA_SUPPORTED_SECONDS: Sequence[int] = (4, 8, 12)
VEO_SUPPORTED_SIZES: Sequence[str] = ("720p", "1080p")
VEO_SUPPORTED_SECONDS: Sequence[int] = (4, 6, 8)

# Model default configuration
# Important note: The core four models (sora2-pro, veo3.1, Wan2.5, ViduQ2) uniformly use 720p resolution (1280x720)
# This facilitates cross-model reference image sharing, for example, using the first frame of veo3.1's first node as reference input for other models
MODEL_DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "sora2-pro": {
        "seconds": 12,
        "size": "1280x720",  # Uniformly use 720p (1280x720) for cross-model reference image sharing
    },
    "sora2": {
        "seconds": 8,
        "size": "1280x720",
    },
    "veo3.1": {
        "seconds": 8,
        "size": "720p",  # Uniformly use 720p (1280x720) for cross-model reference image sharing
    },
    "veo3.1-fast": {
        "seconds": 6,
        "size": "720p",
    },
    "jimeng": {
        "seconds": 10,
        "size": "16:9",  # Aspect ratio format
    },
    "keling": {
        "seconds": 10,
        "size": "16:9",
    },
    "vidu_refer": {
        "seconds": 10,
        "size": "720p",  # Uniformly use 720p (1280x720) for cross-model reference image sharing
    },
    "vidu_image": {
        "seconds": 10,
        "size": "720p",  # Uniformly use 720p (1280x720) for cross-model reference image sharing
    },
    "ViduQ2": {
        "seconds": 10,
        "size": "720p",  # Uniformly use 720p, ViduQ2 automatic mode (first node image, subsequent refer)
    },
    "wan_t2v": {
        "seconds": 10,
        "size": "720P",  # Uniformly use 720p, Wan2.5 text-to-video generation
    },
    "wan_i2v": {
        "seconds": 10,
        "size": "720P",  # Uniformly use 720p, Wan2.5 image-to-video
    },
    "Wan2.5": {
        "seconds": 10,
        "size": "720P",  # Uniformly use 720p, Wan2.5 automatic mode (first node t2v, subsequent i2v)
    },
}


_PARENTHESIS_PATTERNS = (
    re.compile(r"（[^）]*）"),
    re.compile(r"\([^)]*\)"),
    re.compile(r"【[^】]*】"),
)
_DIALOGUE_SPLIT_PATTERN = re.compile(r"[:：]", re.UNICODE)
_NARRATION_PREFIXES = (
    "narrator",
    "narrator:",
    "narrator：",
    "narrator-",
    "narration",
    "voice over",
    "voice-over",
)


def _strip_parenthetical_content(text: str) -> str:
    cleaned = text
    for pattern in _PARENTHESIS_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def _is_narration_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    normalized = stripped.replace("：", ":").lower()
    return any(normalized.startswith(prefix) for prefix in _NARRATION_PREFIXES)


def is_content_moderation_error(error_message: str) -> bool:
    """Check if the error message is related to content moderation.
    
    Args:
        error_message: Error message returned by API
        
    Returns:
        Return True if it's a content moderation error, otherwise return False
    """
    if not error_message:
        return False
    
    error_lower = error_message.lower()
    return any(keyword in error_lower for keyword in CONTENT_MODERATION_KEYWORDS)


def extract_dialogue_text(node_text: str) -> str:
    cleaned = _strip_parenthetical_content(node_text)
    lines = [line.strip() for line in cleaned.splitlines()]
    dialogues: List[str] = []
    for line in lines:
        if not line:
            continue
        if _is_narration_line(line):
            continue
        if not _DIALOGUE_SPLIT_PATTERN.search(line):
            continue
        parts = _DIALOGUE_SPLIT_PATTERN.split(line, maxsplit=1)
        if len(parts) != 2:
            continue
        content = parts[1].strip()
        if content:
            dialogues.append(content)
    return "\n".join(dialogues).strip()


class PaddlespeechTTSDurationEstimator:
    """Estimate text-to-speech duration using PaddleSpeech command-line tool."""

    def __init__(
        self,
        binary: str,
        model_name: str,
        sample_rate: int,
        device: str,
        temp_dir: str,
        speaker_id: Optional[int] = None,
        ffprobe_bin: str = "ffprobe",
        cleanup: bool = True,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.binary = binary
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device
        self.temp_dir = resolve_path(temp_dir)
        self.speaker_id = speaker_id
        self.ffprobe_bin = ffprobe_bin
        self.cleanup = cleanup
        self.extra_env = extra_env or {}
        self._lock = threading.Lock()
        self._counter = 0
        os.makedirs(self.temp_dir, exist_ok=True)

    def estimate_seconds(self, text: str) -> float:
        content = text.strip()
        if not content:
            return 0.0

        with self._lock:
            self._counter += 1
            index = self._counter

        segment_dir = os.path.join(self.temp_dir, f"seg_{index:04d}")
        os.makedirs(segment_dir, exist_ok=True)
        wav_path = os.path.join(segment_dir, "speech.wav")

        cmd: List[str] = [
            self.binary,
            "tts",
            "--input",
            content,
            "--output",
            wav_path,
            "--am",
            self.model_name,
            "--device",
            self.device,
            "--sr",
            str(self.sample_rate),
        ]
        if self.speaker_id is not None:
            cmd.extend(["--spk_id", str(self.speaker_id)])

        try:
            self._run_command(cmd)
            duration = self._probe_duration(wav_path)
        finally:
            if self.cleanup:
                shutil.rmtree(segment_dir, ignore_errors=True)

        return duration

    def _run_command(self, cmd: Sequence[str]) -> None:
        LOGGER.debug("Executing TTS command: %s", " ".join(cmd))
        env = os.environ.copy()
        env.update(self.extra_env)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        if proc.returncode != 0:
            LOGGER.error("TTS command failed: %s", proc.stderr.strip())
            raise RuntimeError(f"TTS command execution failed, return code {proc.returncode}")

    def _probe_duration(self, wav_path: str) -> float:
        if not os.path.exists(wav_path):
            raise RuntimeError(f"No audio file generated: {wav_path}")

        cmd = [
            self.ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            wav_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            LOGGER.error("ffprobe duration parsing failed: %s", proc.stderr.strip())
            raise RuntimeError("ffprobe failed to get audio duration")
        try:
            return float(proc.stdout.strip())
        except ValueError as exc:
            raise RuntimeError("Unable to parse duration from ffprobe output") from exc

def ensure_dependencies() -> None:
    """Validate third-party dependencies to avoid runtime errors."""

    missing: List[str] = []
    if cv2 is None:
        missing.append("opencv-python")
    if VideoFileClip is None or concatenate_videoclips is None:
        missing.append("moviepy")
    if requests is None:
        missing.append("requests")
    if missing:
        raise RuntimeError("Missing required dependencies, please install: {}".format(", ".join(missing)))


def size_str_to_tuple(size: str) -> Optional[Tuple[int, int]]:
    match = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", size)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


@dataclass
class ReferenceImageSpec:
    path: str
    mode: str

    def __post_init__(self) -> None:
        normalized = self.mode.lower()
        if normalized not in REFERENCE_MODE_CHOICES:
            raise ValueError(f"Unsupported reference image mode: {self.mode}")
        self.mode = normalized

    def build_message(self, model: str, size: str) -> Dict[str, str]:
        if cv2 is None:
            raise RuntimeError("opencv-python not found, please install and try again.")
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Reference image does not exist: {self.path}")

        image = cv2.imread(self.path)
        if image is None:
            raise RuntimeError(f"Failed to read reference image: {self.path}")

        resized = image
        if model.startswith("sora"):
            target_size = size_str_to_tuple(size)
            if target_size:
                resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        success, buffer = cv2.imencode(".png", resized)
        if not success:
            raise RuntimeError(f"Image encoding failed: {self.path}")
        image_data = base64.b64encode(buffer.tobytes()).decode("utf-8")

        # Special handling for Wan2.5 and ViduQ2 models: use simple image_url format
        if model in ("Wan2.5", "ViduQ2"):
            return {
                "type": "image_url",
                "value": f"data:image/png;base64,{image_data}",
            }
        
        # Processing for other models
        if self.mode == "first":
            return {
                "type": "image_url",
                "value": f"data:image/png;base64,{image_data}",
            }
        return {
            "type": "reference_image_url",
            "value": f"data:image/png;base64,{image_data}",
            "reference_type": self.mode,
        }


@dataclass
class StoryComponents:
    nodes: List[str]
    characters: str
    scene: str
    station_nodes: List[str]
    dialog_text: str
    time_spans: List[Tuple[int, int]]


def parse_script_nodes(script_text: str) -> List[str]:
    cleaned = script_text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []

    pattern = re.compile(r"\s*(\d+)\.\s*")
    matches = list(pattern.finditer(cleaned))
    if not matches:
        return [cleaned]

    nodes: List[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        content = cleaned[start:end].strip()
        if content:
            nodes.append(content)
    return nodes


SECTION_PATTERN = re.compile(r"【([^】]+)】：?")


def parse_script_sections(script_text: str) -> Dict[str, str]:
    matches = list(SECTION_PATTERN.finditer(script_text))
    print("matches",matches)
    if not matches:
        return {}

    # ✅ Add English name mapping
    name_map = {
        "dialogue": "dialogue",
        "character profiles": "character profiles",
        "scene description": "scene description",
        "blocking": "blocking",
    }

    sections: Dict[str, str] = {}
    for index, match in enumerate(matches):
        section_name = match.group(1).strip()
        
        # ✅ Uniformly convert to Chinese names
        normalized = section_name.lower()
        if normalized in name_map:
            section_name = name_map[normalized]
        
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(script_text)
        sections[section_name] = script_text[start:end].strip()
    
    return sections

def parse_station_nodes(station_text: str, expected: int) -> List[str]:
    if expected <= 0:
        return []
    if not station_text:
        return ["Station information to be supplemented"] * expected
    pattern = re.compile(r"^\s*(\d+)[\.、\-]?\s*(.+)$", re.M)
    entries = [match.group(2).strip() for match in pattern.finditer(station_text)]
    if not entries:
        entries = [item.strip() for item in re.split(r"[\n；;]", station_text) if item.strip()]
    entries = [entry if entry else "Station information to be supplemented" for entry in entries]
    if len(entries) < expected:
        filler = entries[-1] if entries else "Station information to be supplemented"
        entries.extend([filler] * (expected - len(entries)))
    if len(entries) > expected:
        entries = entries[:expected]
    return entries


def extract_time_spans_from_nodes(nodes: List[str]) -> List[Tuple[int, int]]:
    span_pattern = re.compile(r"\[(\d+)seconds-(\d+)seconds\]")
    spans: List[Tuple[int, int]] = []
    for node in nodes:
        match = span_pattern.search(node)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            if end < start:
                start, end = end, start
            spans.append((start, end))
        else:
            spans.append((0, 0))
    return spans


def extract_story_components(script_text: str) -> StoryComponents:
    sections = parse_script_sections(script_text)
    dialog_text = sections.get("dialogue", script_text)
    nodes = parse_script_nodes(dialog_text)

    characters_text = (sections.get("character profiles") or "").strip()
    scene_text = (sections.get("scene description") or "").strip()
    station_text = sections.get("blocking") or ""
    station_nodes = parse_station_nodes(station_text, len(nodes))
    time_spans = extract_time_spans_from_nodes(nodes)

    return StoryComponents(
        nodes=nodes,
        characters=characters_text,
        scene=scene_text,
        station_nodes=station_nodes,
        dialog_text=(dialog_text or "").strip(),
        time_spans=time_spans,
    )


def resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def slugify_model_name(model: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", model)


def prepare_output_structure(base_dir: str, model: str) -> Tuple[str, str, str, str, str]:
    resolved_base = resolve_path(base_dir)
    model_slug = slugify_model_name(model)
    model_dir = os.path.join(resolved_base, model_slug)
    node_root = os.path.join(model_dir, "node_video")
    final_root = os.path.join(model_dir, "final_video")
    os.makedirs(node_root, exist_ok=True)
    os.makedirs(final_root, exist_ok=True)
    responses_path = os.path.join(model_dir, "responses.jsonl")
    return model_dir, node_root, final_root, responses_path, model_slug


def determine_next_index(responses_path: str) -> int:
    if not os.path.exists(responses_path):
        return 1
    max_index = 0
    try:
        with open(responses_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                index_value = payload.get("index")
                if isinstance(index_value, int) and index_value > max_index:
                    max_index = index_value
    except OSError:
        return 1
    return max_index + 1


def scan_existing_videos(final_root: str, model_slug: str, style_slug: Optional[str] = None) -> set:
    """Scan existing final video files and return the set of generated script indices
    
    Args:
        final_root: Path to the final video directory
        model_slug: Model identifier
        style_slug: Style identifier (optional)
    
    Returns:
        Set of script indices for generated videos
    """
    existing_indices = set()
    
    if not os.path.exists(final_root):
        return existing_indices
    
    # Build filename pattern
    if style_slug:
        # Match model_style_XXX.mp4 format
        pattern = re.compile(rf"^{re.escape(model_slug)}_{re.escape(style_slug)}_(\d+)\.mp4$")
    else:
        # Match model_XXX.mp4 format
        pattern = re.compile(rf"^{re.escape(model_slug)}_(\d+)\.mp4$")
    
    LOGGER.info(f"🔍 Scanning existing video files...")
    LOGGER.info(f"   - Directory: {final_root}")
    LOGGER.info(f"   - Match pattern: {pattern.pattern}")
    
    for filename in os.listdir(final_root):
        if not filename.endswith('.mp4'):
            continue
        
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            file_path = os.path.join(final_root, filename)
            file_size = os.path.getsize(file_path)
            
            if file_size > 0:
                existing_indices.add(index)
                LOGGER.info(f"   ✓ Found existing video: {filename} (index={index}, size={file_size / 1024 / 1024:.2f}MB)")
            else:
                LOGGER.warning(f"   ⚠ Found empty video file: {filename}")
    
    if existing_indices:
        LOGGER.info(f"✅ Found {len(existing_indices)} existing videos: {sorted(existing_indices)}")
    else:
        LOGGER.info(f"📝 No existing videos found")
    
    return existing_indices


def append_response_record(
    responses_path: str,
    index: int,
    response_text: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    record: Dict[str, Any] = {
        "index": index,
        "response": response_text,
    }
    if metadata:
        record["meta"] = metadata
    with open(responses_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False))
        file.write("\n")


def append_video_dialogue_records(output_path: str, mapping: Dict[str, str]) -> None:
    if not mapping:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as file:
        for filename in sorted(mapping.keys()):
            record = {filename: mapping[filename]}
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")


def iter_batch_responses(jsonl_path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    resolved = resolve_path(jsonl_path)
    with open(resolved, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:  # pragma: no cover - error tolerance
                LOGGER.warning("JSONL line %d parsing failed: %s", line_no, exc)
                continue
            yield line_no, payload


def update_jsonl_line(jsonl_path: str, line_no: int, updated_response: str) -> None:
    """Update the response field of the specified line in the JSONL file (script after removing sensitive nodes)"""
    
    # Read all lines
    resolved = resolve_path(jsonl_path)
    with open(resolved, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # Update the specified line
    updated = False
    current_line = 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        current_line += 1
        if current_line == line_no:
            try:
                payload = json.loads(line.strip())
                payload["response"] = updated_response
                lines[i] = json.dumps(payload, ensure_ascii=False) + "\n"
                updated = True
                LOGGER.info(f"✅ Updated {jsonl_path} line {line_no} (script after removing sensitive nodes)")
                break
            except json.JSONDecodeError:
                LOGGER.error(f"Failed to parse line {line_no}, update failed")
                return
    
    if not updated:
        LOGGER.warning(f"Line {line_no} not found, update failed")
        return
    
    # Write back to file
    with open(resolved, "w", encoding="utf-8") as file:
        file.writelines(lines)


def _rebuild_script_with_original_format(
    original_script: str, 
    updated_nodes: List[str], 
    updated_stations: List[str]
) -> str:
    """
    Rebuild script, maintaining original format and section labels
    
    Args:
        original_script: Original script text
        updated_nodes: Updated dialogue node list
        updated_stations: Updated station node list
    
    Returns:
        Rebuilt script text
    """
    # Detect the label format used in the original script
    section_markers = list(SECTION_PATTERN.finditer(original_script))
    if not section_markers:
        # No section markers, directly return dialogue nodes
        return "\n".join(f"{i+1}. {node}" for i, node in enumerate(updated_nodes))
    
    # Extract original section information (maintaining original label names and order)
    original_sections = []
    for index, match in enumerate(section_markers):
        section_name = match.group(1).strip()  # Maintain original labels
        start = match.end()
        end = section_markers[index + 1].start() if index + 1 < len(section_markers) else len(original_script)
        content = original_script[start:end].strip()
        original_sections.append((section_name, content))
    
    # Label mapping (for identification)
    label_map = {
        "dialogue": "dialogue",
        "character profiles": "character profiles",
        "scene description": "scene description",
        "blocking": "blocking",
    }
    
    # Rebuild script, maintaining original order and label format
    rebuilt_parts = []
    for section_name, original_content in original_sections:
        # Normalize for comparison (convert to lowercase)
        normalized_name = section_name.lower()
        
        # Determine if it's a dialogue section
        if normalized_name in ("dialogue", "dialogue"):
            # ✅ Update dialogue content
            updated_dialog = "\n".join(f"{i+1}. {node}" for i, node in enumerate(updated_nodes))
            rebuilt_parts.append(f"【{section_name}】：\n{updated_dialog}")
        # Determine if it's a blocking section
        elif normalized_name in ("blocking", "blocking"):
            # ✅ Update blocking content
            if updated_stations:
                updated_blocking = "\n".join(f"{i+1}. {station}" for i, station in enumerate(updated_stations))
                rebuilt_parts.append(f"【{section_name}】：\n{updated_blocking}")
            else:
                # If no station information, skip this section
                pass
        else:
            # ✅ Keep other sections unchanged (character profiles, scene description, etc.)
            rebuilt_parts.append(f"【{section_name}】：\n{original_content}")
    
    return "\n\n".join(rebuilt_parts)

def process_story(
    generator: "StoryVideoGenerator",
    script_text: str,
    model_slug: str,
    story_index: int,
    node_root: str,
    final_root: str,
    responses_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    components: Optional[StoryComponents] = None,
    style_slug: Optional[str] = None,
    dialogue_map: Optional[Dict[str, str]] = None,
    resume_node_dir: Optional[str] = None,
    source_jsonl: Optional[str] = None,  # ✅ Added: Source JSONL path
    source_line_no: Optional[int] = None,  # ✅ Added: Source JSONL line number
) -> str:
    if components is None:
        components = extract_story_components(script_text)
    nodes = components.nodes
    if not nodes:
        raise ValueError("Failed to parse valid nodes, please check if the script text format is 'number. content'.")

    script_stub = f"{model_slug}_{story_index:03d}"
    if style_slug:
        script_stub = f"{model_slug}_{style_slug}_{story_index:03d}"
    
    final_video_path = os.path.join(final_root, f"{script_stub}.mp4")
    
    # ✅ Add detailed logs
    LOGGER.info(f"🔍 Detection path information:")
    LOGGER.info(f"   - model_slug: {model_slug}")
    LOGGER.info(f"   - style_slug: {style_slug}")
    LOGGER.info(f"   - story_index: {story_index}")
    LOGGER.info(f"   - script_stub: {script_stub}")
    LOGGER.info(f"   - final_root: {final_root}")
    LOGGER.info(f"   - final_video_path: {final_video_path}")
    LOGGER.info(f"   - File exists: {os.path.exists(final_video_path)}")
    
    # ✅ Mechanism 2: Auto-resume - Check if final video already exists
    if os.path.exists(final_video_path):
        file_size = os.path.getsize(final_video_path)
        if file_size > 0:
            LOGGER.info(f"🎬 Script #{story_index} final video already exists, skipping generation: {final_video_path} ({file_size / 1024 / 1024:.2f} MB)")
            # Still record dialogue information
            if dialogue_map is not None:
                dialogue_map[os.path.basename(final_video_path)] = components.dialog_text.strip()
            return final_video_path
        else:
            LOGGER.warning(f"Detected empty final video file, will regenerate: {final_video_path}")
            os.remove(final_video_path)
    
    # ✅ Continue generation mode: Use specified node directory
    if resume_node_dir:
        node_output_dir = resolve_path(resume_node_dir)
        LOGGER.info(f"Continue generation mode: Using node directory {node_output_dir}")
    else:
        node_output_dir = os.path.join(node_root, script_stub)
        if os.path.exists(node_output_dir):
            shutil.rmtree(node_output_dir)
        os.makedirs(node_output_dir, exist_ok=True)

    # ✅ Detect generated nodes (for continue generation mode)
    resume_from = 1
    resume_reference_path = None
    
    if resume_node_dir and os.path.exists(node_output_dir):
        existing_videos, next_index, last_video = detect_existing_nodes(node_output_dir)
        if existing_videos:
            resume_from = next_index
            LOGGER.info(f"🔄 Detected {len(existing_videos)} generated nodes, will continue from node {resume_from}")
            if last_video:
                # Extract reference frame from the last video
                last_basename = os.path.splitext(os.path.basename(last_video))[0]
                resume_reference_path = os.path.join(
                    node_output_dir,
                    f"{last_basename}_last_frame.png"
                )
                if not os.path.exists(resume_reference_path):
                    LOGGER.info("Extracting reference frame from the last node...")
                    extract_last_frame(last_video, resume_reference_path)

    LOGGER.info("Script #%d parsed %d nodes, starting generation.", story_index, len(nodes))

    final_nodes, final_stations, nodes_deleted = generator.generate(
        nodes,
        final_video_path=final_video_path,
        node_output_dir=node_output_dir,
        characters_text=components.characters,
        scene_text=components.scene,
        station_nodes=components.station_nodes,
        time_spans=components.time_spans,
        style_slug=style_slug,
        resume_from=resume_from,  # ✅ Pass starting node
        resume_reference_path=resume_reference_path,  # ✅ Pass reference image
    )
    
    # ✅ If nodes were deleted, update the source JSONL file
    if nodes_deleted and source_jsonl and source_line_no:
        LOGGER.warning(f"⚠️ Script #{story_index} has nodes deleted, updating source JSONL file...")
        
        # ✅ Smart rebuild script: maintain original section label format (English)
        updated_script = _rebuild_script_with_original_format(
            script_text,
            final_nodes,
            final_stations
        )
        update_jsonl_line(source_jsonl, source_line_no, updated_script)
        LOGGER.info(f"✅ Updated dialogue nodes({len(final_nodes)}) and blocking nodes({len(final_stations)})")

    append_response_record(responses_path, story_index, script_text, metadata)
    LOGGER.info("Script #%d processing completed, final video: %s", story_index, final_video_path)
    if dialogue_map is not None:
        dialogue_map[os.path.basename(final_video_path)] = components.dialog_text.strip()
    return final_video_path

def safe_filename(text: str, default: str = "segment") -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z]+", "_", text)[:40]
    sanitized = sanitized.strip("_")
    return sanitized or default
def _extract_frame_ffmpeg(video_path: str, frame_output_path: str) -> str:
    """Extract last frame using ffmpeg (fixed version - absolutely reliable)"""
    import subprocess
    
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found")
    
    os.makedirs(os.path.dirname(frame_output_path), exist_ok=True)
    
    probe_cmd = [
        shutil.which("ffprobe") or "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        probe_result = subprocess.run(
            probe_cmd, 
            capture_output=True, 
            text=True, 
            timeout=10, 
            check=True
        )
        duration = float(probe_result.stdout.strip())
        LOGGER.info(f"📹 Video exact duration: {duration:.3f}s")
    except Exception as e:
        LOGGER.warning(f"Failed to get exact duration, using fallback method: {e}")
        duration = None
    
    # ✅ Method 1: Use exact timestamp (most reliable)
    if duration is not None:
        # Extract position 0.1 seconds before the end
        seek_time = max(0, duration - 0.1)
        cmd = [
            ffmpeg_bin,
            "-accurate_seek",        # Accurate positioning
            "-ss", f"{seek_time:.3f}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "1",
            "-y",
            frame_output_path
        ]
        
        LOGGER.info(f"Using ffmpeg to extract last frame (timestamp: {seek_time:.3f}s)...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            
            if result.returncode == 0 and os.path.exists(frame_output_path):
                file_size = os.path.getsize(frame_output_path)
                if file_size > 0:
                    # Verify image
                    if cv2 is not None:
                        test_img = cv2.imread(frame_output_path)
                        if test_img is not None:
                            import hashlib
                            with open(frame_output_path, 'rb') as f:
                                file_hash = hashlib.md5(f.read()).hexdigest()[:8]
                            LOGGER.info(f"✅ ffmpeg method 1 successful: {test_img.shape[1]}x{test_img.shape[0]}, "
                                       f"{file_size / 1024:.2f}KB, hash:{file_hash}")
                            return frame_output_path
            
            LOGGER.warning(f"Method 1 failed: {result.stderr[:200]}")
        except Exception as e:
            LOGGER.warning(f"Method 1 exception: {e}")
    
    # ✅ Method 2: Use reverse lookup (fallback)
    LOGGER.info("Trying ffmpeg method 2 (reverse lookup)...")
    cmd = [
        ffmpeg_bin,
        "-i", video_path,
        "-vf", "select='eq(n\,0)+eq(n\,1)+eq(n\,2)'",  # Select first 3 frames
        "-vsync", "vfr",
        "-frames:v", "1",
        "-q:v", "1",
        "-y",
        frame_output_path
    ]
    
    # Use -sseof to locate from the end
    cmd_reverse = [
        ffmpeg_bin,
        "-sseof", "-1",  # 1 second from the end
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "1",
        "-y",
        frame_output_path
    ]
    
    for attempt_cmd in [cmd_reverse, cmd]:
        try:
            result = subprocess.run(
                attempt_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            
            if result.returncode == 0 and os.path.exists(frame_output_path):
                file_size = os.path.getsize(frame_output_path)
                if file_size > 0:
                    if cv2 is not None:
                        test_img = cv2.imread(frame_output_path)
                        if test_img is not None:
                            import hashlib
                            with open(frame_output_path, 'rb') as f:
                                file_hash = hashlib.md5(f.read()).hexdigest()[:8]
                            LOGGER.info(f"✅ ffmpeg fallback method successful: {test_img.shape[1]}x{test_img.shape[0]}, "
                                       f"{file_size / 1024:.2f}KB, hash:{file_hash}")
                            return frame_output_path
        except Exception as e:
            LOGGER.warning(f"ffmpeg fallback method exception: {e}")
            continue
    
    raise RuntimeError("All ffmpeg methods failed")


def _extract_frame(video_path: str, frame_index: int, frame_output_path: str) -> str:
    """Extract specified frame, try methods in order of reliability"""
    import time
    
    # Wait for file to be fully written
    time.sleep(1.0)
    
    last_error = None
    
    # ✅ Method 1: OpenCV (most stable, preferred)
    if cv2 is not None:
        try:
            LOGGER.info("🔹 Trying method 1: OpenCV")
            return _extract_frame_opencv(video_path, frame_index, frame_output_path)
        except Exception as exc:
            last_error = f"OpenCV failed: {exc}"
            LOGGER.warning(last_error)
    
    # ✅ Method 2: ffmpeg (fallback)
    try:
        LOGGER.info("🔹 Trying method 2: ffmpeg")
        return _extract_frame_ffmpeg(video_path, frame_output_path)
    except Exception as exc:
        last_error = f"ffmpeg failed: {exc}"
        LOGGER.warning(last_error)
    
    # ✅ Method 3: moviepy (last resort)
    if VideoFileClip is not None:
        try:
            LOGGER.info("🔹 Trying method 3: moviepy")
            return _extract_frame_moviepy(video_path, frame_output_path)
        except Exception as exc:
            last_error = f"moviepy failed: {exc}"
            LOGGER.warning(last_error)
    
    raise RuntimeError(f"All methods failed: {last_error}")

def extract_last_frame(video_path: str, frame_output_path: str) -> str:
    """Extract last frame of video (entry function - enhanced version)"""
    import time
    import subprocess
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    
    # ✅ Step 1: Wait for file size to stabilize (ensure writing is complete)
    LOGGER.info("⏳ Waiting for video file to be written completely...")
    max_wait = 30  # Wait at most 30 seconds
    check_interval = 1.0
    stable_count = 0
    required_stable = 3  # Consider stable after 3 consecutive size checks
    
    last_size = 0
    for i in range(int(max_wait / check_interval)):
        current_size = os.path.getsize(video_path)
        
        if current_size == 0:
            LOGGER.warning(f"Check {i+1}: File size is 0, continuing to wait...")
            time.sleep(check_interval)
            continue
        
        if current_size == last_size:
            stable_count += 1
            LOGGER.debug(f"File size stable ({stable_count}/{required_stable}): {current_size / 1024 / 1024:.2f} MB")
            if stable_count >= required_stable:
                break
        else:
            stable_count = 0
            LOGGER.debug(f"File size changed: {last_size / 1024 / 1024:.2f} -> {current_size / 1024 / 1024:.2f} MB")
        
        last_size = current_size
        time.sleep(check_interval)
    
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        raise RuntimeError(f"Video file size is 0, might have failed to write")
    
    LOGGER.info(f"✅ File writing complete: {file_size / 1024 / 1024:.2f} MB")
    
    # ✅ Step 2: Force sync filesystem (Linux only)
    try:
        if hasattr(os, 'sync'):
            os.sync()
            LOGGER.debug("Executed os.sync()")
    except Exception as e:
        LOGGER.debug(f"os.sync() failed: {e}")
    
    # ✅ Step 3: Verify video readability
    LOGGER.info("🔍 Verifying video file integrity...")
    try:
        # Use ffprobe to check video metadata
        ffprobe_bin = shutil.which("ffprobe")
        if ffprobe_bin:
            cmd = [
                ffprobe_bin,
                "-v", "error",
                "-select_streams", "v:0",
                "-count_frames",
                "-show_entries", "stream=nb_read_frames,duration,r_frame_rate",
                "-of", "json",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                if info.get('streams'):
                    stream = info['streams'][0]
                    frame_count = stream.get('nb_read_frames', 'unknown')
                    duration = stream.get('duration', 'unknown')
                    fps = stream.get('r_frame_rate', 'unknown')
                    LOGGER.info(f"📹 Video info: frames={frame_count}, duration={duration}s, FPS={fps}")
            else:
                LOGGER.warning(f"ffprobe check failed: {result.stderr}")
    except Exception as e:
        LOGGER.warning(f"Video integrity check failed: {e}")
    
    # ✅ Step 4: Extra wait (CephFS may need more time)
    time.sleep(2.0)
    
    LOGGER.info(f"Preparing to extract reference frame - file: {os.path.basename(video_path)}")
    
    # Calculate target frame index
    frame_index = 0
    if cv2 is not None:
        capture = cv2.VideoCapture(video_path)
        if capture.isOpened():
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = capture.get(cv2.CAP_PROP_FPS) or 30
            duration = total_frames / fps if fps > 0 and total_frames > 0 else 0
            LOGGER.info(f"OpenCV read - total frames: {total_frames}, FPS: {fps:.2f}, duration: {duration:.2f}s")
            # ✅ Use total_frames - 1 (true last frame index)
            frame_index = max(total_frames - 1, 0) if total_frames > 0 else 0
            capture.release()
    
    return _extract_frame(video_path, frame_index, frame_output_path)

def _extract_frame_moviepy(video_path: str, frame_output_path: str) -> str:
    """Extract last frame using moviepy (improved version)"""
    if VideoFileClip is None or cv2 is None:
        raise RuntimeError("moviepy or OpenCV not available")
    
    clip = VideoFileClip(video_path)
    try:
        duration = float(clip.duration or 0.0)
        fps = float(clip.fps or 0.0)
        
        if duration <= 0 or fps <= 0:
            raise RuntimeError(f"Video duration({duration}) or FPS({fps}) abnormal")
        
        # ✅ Improvement: Use duration - 0.01 seconds directly
        target_time = max(0, duration - 0.01)
        LOGGER.info("Video total duration: %.3f seconds, extraction position: %.3f seconds", duration, target_time)
        
        frame_rgb = clip.get_frame(target_time)
        
        if frame_rgb is None or frame_rgb.size == 0:
            raise RuntimeError("moviepy did not return a valid frame")
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Save
        os.makedirs(os.path.dirname(frame_output_path), exist_ok=True)
        if not cv2.imwrite(frame_output_path, frame_bgr):
            raise RuntimeError("Failed to save image")
        
        LOGGER.info("✓ moviepy extraction successful (%.3fs position, %dx%d)", 
                   target_time, frame_bgr.shape[1], frame_bgr.shape[0])
        return frame_output_path
        
    finally:
        clip.close()

def _extract_frame_opencv(video_path: str, frame_index: int, frame_output_path: str) -> str:
    """Extract specified frame using OpenCV (enhanced version)"""
    if cv2 is None:
        raise RuntimeError("OpenCV not available")
    
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("OpenCV cannot open video")
    
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        
        if total_frames <= 0:
            raise RuntimeError("Video has 0 frames")
        
        LOGGER.info(f"OpenCV reading video - total frames: {total_frames}, FPS: {fps:.2f}")
        
        # ✅ Try multiple candidate frames from the end
        candidates = [
            total_frames - 1,   # Last frame
            total_frames - 2,   # 2nd last frame
            total_frames - 5,   # 5th last frame
            total_frames - 10,  # 10th last frame
            int(total_frames * 0.95),  # 95% position
        ]
        
        # Ensure indices are valid
        candidates = [max(0, min(idx, total_frames - 1)) for idx in candidates]
        
        frame = None
        used_index = -1
        
        for candidate_idx in candidates:
            # ✅ Method 1: Directly set frame position
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)
            
            success, temp_frame = capture.read()
            if success and temp_frame is not None and temp_frame.size > 0:
                # Verify frame is not all black
                gray = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)
                brightness = gray.mean()
                
                if brightness > 5:  # Brightness threshold
                    frame = temp_frame
                    used_index = candidate_idx
                    LOGGER.info(f"✅ OpenCV extracted frame {candidate_idx}/{total_frames} (brightness: {brightness:.2f})")
                    break
                else:
                    LOGGER.warning(f"Frame {candidate_idx} brightness too low({brightness:.2f}), trying next")
            else:
                LOGGER.warning(f"Frame {candidate_idx} read failed")
        
        if frame is None:
            # ✅ Last resort: Read frame by frame to the end
            LOGGER.warning("All candidate frames failed, trying frame by frame read...")
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            last_valid_frame = None
            last_valid_index = 0
            
            for i in range(total_frames):
                success, temp_frame = capture.read()
                if success and temp_frame is not None and temp_frame.size > 0:
                    last_valid_frame = temp_frame
                    last_valid_index = i
            
            if last_valid_frame is not None:
                frame = last_valid_frame
                used_index = last_valid_index
                LOGGER.info(f"✅ Frame by frame read successful, using frame {used_index}")
        
        if frame is None:
            raise RuntimeError("All methods failed to extract valid frame")
        
        # Save
        os.makedirs(os.path.dirname(frame_output_path), exist_ok=True)
        if not cv2.imwrite(frame_output_path, frame):
            raise RuntimeError("Failed to save image")
        
        file_size = os.path.getsize(frame_output_path)
        LOGGER.info(f"✅ OpenCV save successful: {frame.shape[1]}x{frame.shape[0]}, {file_size / 1024:.2f}KB")
        
        return frame_output_path
        
    finally:
        capture.release()

def stitch_videos(video_paths: List[str], output_path: str) -> None:
    if not video_paths:
        raise ValueError("No video files provided for concatenation.")

    if VideoFileClip is None or concatenate_videoclips is None:
        raise RuntimeError("moviepy not found, please install it first.")

    clips = []
    try:
        for path in video_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video file does not exist: {path}")
            clips.append(VideoFileClip(path))
        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            remove_temp=True,
        )
    finally:
        for clip in clips:
            clip.close()


def get_simple_auth(source: str, secret_id: str, secret_key: str) -> Tuple[str, str]:
    date_time = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    auth_prefix = (
        "hmac id=\"{}\", algorithm=\"hmac-sha1\", headers=\"date source\", signature=\"".format(secret_id)
    )
    sign_str = "date: {}\nsource: {}".format(date_time, source)
    digest = hmac.new(secret_key.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha1).digest()
    signature = base64.b64encode(digest).decode("utf-8")
    return auth_prefix + signature + "\"", date_time


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class Api:
    """Unified client for calling video generation APIs (supports multiple models)"""

    def __init__(self, user: str, apikey: str) -> None:
        LOGGER.info("Initializing API client, account: %s", user)
        HOST = "trpc-gpt-eval.production.polaris"
        self.host = "http://{}:8080".format(HOST)
        self.user = user
        self.apikey = apikey
        self.timeout = 3600
        self.models = {
            "sora2-pro": "api_openai_sora-2-pro",
            "sora2": "api_openai_sora-2",
            "veo3.1": "api_google_veo-3.1-generate-preview",
            "veo3.1-fast": "api_google_veo-3.1-fast-generate-preview",
            "jimeng": "api_doubao_jimeng_ti2v_v30_pro",
            "keling": "api_klingai_text2video:kling-v2-5-turbo",
            "vidu_refer": "api_vidu_reference-to-video_viduq2",
            "vidu_image": "api_vidu_img2video_viduq2-pro",
            # "wan_t2v": "api_ali_wan2.5-t2v-preview",
            # "wan_i2v": "api_ali_wan2.5-i2v-preview",
            "wan_t2v": "api_ali_wan2.6-t2v",
            "wan_i2v": "api_ali_wan2.6-i2v",
            "Wan2.5": "api_ali_wan2.6-i2v",  # Placeholder, will switch dynamically
            "ViduQ2": "api_vidu_reference-to-video_viduq2",  # Placeholder, will switch dynamically
        }

    def get_header(self) -> Dict[str, str]:
        source = "VideoEvaluate"
        sign, date_time = get_simple_auth(source, self.user, self.apikey)
        return {
            "Apiversion": API_VERSION,
            "Authorization": sign,
            "Date": date_time,
            "Source": source,
        }
    def call_data_eval(
        self,
        text: str,
        pic_path: Optional[List[str]],
        pic_func: Optional[List[str]],
        output_path: str,
        model: str,
        size: str,
        seconds: int,
        **kargs: Any,
    ) -> Optional[str]:
        if model not in self.models:
            raise ValueError("Unknown model: {}".format(model))
        if requests is None:
            raise RuntimeError("requests not found, please install it first.")

        base_url = self.host + "/api/v1/data_eval"

        # Build request data
        data: Dict[str, Any] = {
            "request_id": str(uuid.uuid4()),
            "model_marker": self.models[model],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "value": text,
                        }
                    ],
                }
            ],
            "timeout": self.timeout,
        }

        # Set parameters based on model type
        params: Dict[str, Any] = {}
        
        if model.startswith("sora"):
            params = {
                "seconds": seconds,
                "size": size,
            }
        elif model.startswith("veo"):
            params = {
                "durationSeconds": seconds,
                "resolution": size,
                "aspectRatio": "16:9",
                "generateAudio": True,
            }
        elif model.startswith("jimeng"):
            params = {
                "aspectRatio": size,
                "frames": seconds * 24 + 1,
            }
        elif model.startswith("keling"):
            params = {
                "aspect_ratio": size,
                "duration": seconds,
            }
        elif model.startswith("vidu"):
            params = {
                "duration": seconds,
                "resolution": size,
                "bgm": True,
            }
        elif model.startswith("wan"):
            params = {
                "duration": seconds,
                "resolution": size,
                "audio": True,
            }
        
        # Update extra parameters
        params.update(kargs)
        data["params"] = params

        LOGGER.info(f"Text length: {len(text)}")

        # Process images
        if pic_path is not None and pic_func is not None:
            if cv2 is None:
                raise RuntimeError("opencv-python not found, please install it first.")
            
            for sub_pic_path, sub_pic_func in zip(pic_path, pic_func):
                img = cv2.imread(sub_pic_path)
                if img is None:
                    raise RuntimeError(f"Cannot read image: {sub_pic_path}")
                
                # Sora model needs image resizing
                if model.startswith("sora"):
                    new_size = size.split("x")
                    new_size = (int(new_size[0]), int(new_size[1]))
                    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
                else:
                    resized_img = img
                
                LOGGER.info(f"Image size: {resized_img.shape}")
                retval, buffer = cv2.imencode(".png", resized_img)
                if not retval:
                    raise RuntimeError(f"Image encoding failed: {sub_pic_path}")
                
                img_data = base64.b64encode(buffer.tobytes()).decode("utf-8")
                
                # Add images based on model and function type
                if model.startswith("jimeng"):
                    data["params"]["binary_data_base64"] = [img_data]
                elif model.startswith("keling"):
                    data["messages"][0]["content"].append({
                        "type": "image_url",
                        "value": img_data,
                    })
                elif model.startswith("vidu") or model.startswith("wan"):
                    data["messages"][0]["content"].append({
                        "type": "image_url",
                        "value": f"data:image/png;base64,{img_data}",
                    })
                elif sub_pic_func == "first":
                    data["messages"][0]["content"].append({
                        "type": "image_url",
                        "value": f"data:image/png;base64,{img_data}",
                    })
                elif sub_pic_func == "style":
                    data["messages"][0]["content"].append({
                        "type": "reference_image_url",
                        "value": f"data:image/png;base64,{img_data}",
                        "reference_type": "style",
                    })
                elif sub_pic_func == "asset":
                    data["messages"][0]["content"].append({
                        "type": "reference_image_url",
                        "value": f"data:image/png;base64,{img_data}",
                        "reference_type": "asset",
                    })
                else:
                    raise ValueError(f"Unsupported image function type: {sub_pic_func}")
                
                LOGGER.info(f"Image path: {sub_pic_path}, Image function: {sub_pic_func}")

        headers = dict(self.get_header())
        LOGGER.debug("Request URL: %s", base_url)
        LOGGER.debug("Request headers: %s", headers)

        try:
            rsp = requests.post(url=base_url, headers=headers, json=data, timeout=self.timeout)
            result = rsp.json()
            
            if "answer" in result:
                if result["answer"] is None:
                    msg = result.get("msg", "Unknown error")
                    LOGGER.error(f"API returned error: {msg}")
                    return msg
                else:
                    video_data = result["answer"][0]["value"]
                    
                    # Some models return download links
                    if model in ["jimeng", "keling", "vidu_refer", "vidu_image"] or model.startswith("wan"):
                        LOGGER.info(f"Returned link: {video_data}")
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        import subprocess
                        subprocess.run(
                            ["curl", video_data, "--output", output_path],
                            check=True,
                            capture_output=True,
                        )
                        LOGGER.info(f"Video downloaded to: {output_path}")
                    else:
                        # Other models return base64 encoded video
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        with open(output_path, "wb") as f:
                            f.write(base64.b64decode(video_data))
                        LOGGER.info(f"Video saved to: {output_path}")
                    
                    # Save response metadata
                    metadata_path = os.path.splitext(output_path)[0] + "_response.json"
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    
                    return None
            else:
                error_msg = result.get("msg", json.dumps(result, ensure_ascii=False))
                LOGGER.error(f"API returned exception: {error_msg}")
                return error_msg
                
        except Exception as e:
            LOGGER.exception(f"API call failed: {e}")
            return str(e)


def detect_existing_nodes(node_dir: str) -> Tuple[List[str], int, Optional[str]]:
    """
    Detect existing node videos
    
    Returns:
        - existing_videos: List of existing video file paths (sorted by index)
        - next_node_index: Next node index to generate (starting from 1)
        - last_video_path: Path of the last video, used for extracting reference frame
    """
    if not os.path.exists(node_dir):
        return [], 1, None
    
    # Find all .mp4 files
    video_files = []
    for filename in os.listdir(node_dir):
        if filename.endswith('.mp4') and not filename.startswith('.'):
            full_path = os.path.join(node_dir, filename)
            video_files.append(full_path)
    
    if not video_files:
        return [], 1, None
    
    # Sort by filename (assuming filename contains index)
    video_files.sort()
    
    # Extract maximum index from filename
    max_index = 0
    pattern = re.compile(r'(\d+)')
    for filename in video_files:
        basename = os.path.basename(filename)
        match = pattern.search(basename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
    
    next_index = max_index + 1
    last_video = video_files[-1]
    
    LOGGER.info(f"Detected {len(video_files)} existing node videos")
    LOGGER.info(f"Last node index: {max_index}")
    LOGGER.info(f"Next node index: {next_index}")
    LOGGER.info(f"Last video: {os.path.basename(last_video)}")
    
    return video_files, next_index, last_video

class StoryVideoGenerator:
    """Responsible for script splitting, calling API to generate node videos, and synthesizing the final video."""

    def __init__(
        self,
        api: Api,
        output_dir: str,
        model: str,
        size: str,
        seconds: int,
        max_retry: int,
        reference_mode: str,
        style_key: str,
        veo_aspect: Optional[str] = None,
        veo_audio: Optional[bool] = None,
        tts_estimator: Optional[PaddlespeechTTSDurationEstimator] = None,
        duration_padding: float = 1.5,
        reference_first_frame: Optional[str] = None,
    ) -> None:
        self.api = api
        self.output_dir = resolve_path(output_dir)
        self.model = model
        self.size = size
        self.seconds = seconds
        self.max_retry = max_retry
        if reference_mode not in REFERENCE_MODE_CHOICES:
            raise ValueError(f"Unknown reference image mode: {reference_mode}")
        self.reference_mode = reference_mode
        self.veo_aspect = veo_aspect
        self.veo_audio = veo_audio
        self.style_key = style_key
        self.tts_estimator = tts_estimator
        self.duration_padding = max(0.0, duration_padding)
        self.reference_first_frame = reference_first_frame

        os.makedirs(self.output_dir, exist_ok=True)
        self._reference_spec: Optional[ReferenceImageSpec] = None

    def _build_prompt(
        self,
        node_text: str,
        index: int,
        total: int,
        characters_text: str,
        scene_text: str,
        station_text: str,
        node_seconds: int,
        time_span: Optional[Tuple[int, int]] = None,
    ) -> str:
        segments = [
            CONTINUITY_PROMPT,
            STYLE_PROMPTS[self.style_key],
            f"Shot Number: {index}/{total}.",
        ]
        
        # Add clear duration instructions
        if time_span is not None and time_span[1] > time_span[0]:
            # Has timestamp info, explain the time range from original script
            segments.append(
                f"Important Note: This is an independent new video clip (not a segment of the entire video). "
                f"The original script marks the time range as [{time_span[0]}s-{time_span[1]}s], "
                f"the actual video duration to generate is {time_span[1] - time_span[0]} seconds "
                f"(i.e., {time_span[1]}-{time_span[0]}={time_span[1] - time_span[0]}s), "
                f"and the video timeline starts from 0 seconds."
            )
        else:
            # No timestamp info, directly state video duration
            segments.append(
                f"Important Note: This is an independent new video clip, video duration is {node_seconds} seconds, timeline starts from 0 seconds."
            )
        
        segments.append(f"Shot Script: {node_text.strip()}")
        
        if characters_text.strip():
            segments.append(f"Character Profiles: {characters_text.strip()}")
        if scene_text.strip():
            segments.append(f"Scene Description: {scene_text.strip()}")
        if station_text.strip():
            segments.append(f"Blocking: {station_text.strip()}")
        return "\n".join(segment for segment in segments if segment)
    
    def _update_reference(self, video_path: str, filename_stub: str, base_dir: str) -> ReferenceImageSpec:
        reference_path = os.path.join(base_dir, f"{filename_stub}_last_frame.png")
        
        LOGGER.info("=" * 60)
        LOGGER.info(f"🎬 Starting to extract reference frame: {os.path.basename(video_path)}")
        
        # Record video file information
        video_size = os.path.getsize(video_path) / 1024 / 1024
        LOGGER.info(f"📦 Video file size: {video_size:.2f} MB")
        
        # ✅ Also extract first frame for comparison (for debugging)
        first_frame_path = os.path.join(base_dir, f"{filename_stub}_first_frame_debug.png")
        try:
            if cv2 is not None:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    success, first_frame = cap.read()
                    if success and first_frame is not None:
                        cv2.imwrite(first_frame_path, first_frame)
                        LOGGER.debug(f"Saved first frame (for debugging): {os.path.basename(first_frame_path)}")
                    cap.release()
        except Exception as e:
            LOGGER.debug(f"Failed to extract first frame (does not affect main process): {e}")
        
        # ✅✅✅ Key: Call extract_last_frame to extract the last frame
        extract_last_frame(video_path, reference_path)
        
        # Verify extraction result
        if cv2 is not None and os.path.exists(reference_path):
            img = cv2.imread(reference_path)
            if img is not None:
                import hashlib
                with open(reference_path, 'rb') as f:
                    img_hash = hashlib.md5(f.read()).hexdigest()[:8]
                
                # Calculate image brightness
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness = gray.mean()
                
                # ✅ If first frame exists, compare both
                if os.path.exists(first_frame_path):
                    first_img = cv2.imread(first_frame_path)
                    if first_img is not None:
                        with open(first_frame_path, 'rb') as f:
                            first_hash = hashlib.md5(f.read()).hexdigest()[:8]
                        
                        # Calculate difference
                        diff = cv2.absdiff(img, first_img)
                        diff_score = diff.mean()
                        
                        LOGGER.info(f"🔍 First frame vs Last frame comparison:")
                        LOGGER.info(f"   - First frame MD5: {first_hash}")
                        LOGGER.info(f"   - Last frame MD5: {img_hash}")
                        LOGGER.info(f"   - Difference score: {diff_score:.2f} (higher means more different)")
                        
                        if img_hash == first_hash:
                            LOGGER.error("❌❌❌ Critical warning: Last frame is identical to first frame!")
                        elif diff_score < 5:
                            LOGGER.warning(f"⚠️  Warning: Two frames have very small difference ({diff_score:.2f}), extraction may be wrong")
                
                LOGGER.info(f"🖼️  Reference frame information:")
                LOGGER.info(f"   - Size: {img.shape[1]}x{img.shape[0]}")
                LOGGER.info(f"   - MD5: {img_hash}")
                LOGGER.info(f"   - Brightness: {brightness:.2f}")
                LOGGER.info(f"   - Path: {os.path.basename(reference_path)}")
                
                if brightness < 10:
                    LOGGER.warning("⚠️  Image brightness too low, may be black screen or extraction failed!")
            else:
                LOGGER.error(f"❌ Cannot read extracted image: {reference_path}")
        
        LOGGER.info("=" * 60)
        return ReferenceImageSpec(path=reference_path, mode=self.reference_mode)

    def _validate_params(self) -> None:
        # Wan2.5 series model parameter validation (including Wan2.5, wan_t2v, wan_i2v)
        if self.model in ("Wan2.5", "wan_t2v", "wan_i2v"):
            LOGGER.info(f"{self.model} model using size: {self.size}, duration: {self.seconds} seconds")
            # Wan2.5 supports 480P, 720P, 1080P formats, parameters are flexible
        # Vidu series model parameter validation
        elif self.model in ("ViduQ2", "vidu_refer", "vidu_image"):
            LOGGER.info(f"{self.model} model using size: {self.size}, duration: {self.seconds} seconds")
            # Vidu series supports flexible parameters
        # Jimeng and Keling models
        elif self.model in ("jimeng", "keling"):
            LOGGER.info(f"{self.model} model using size: {self.size}, duration: {self.seconds} seconds")
            # Aspect ratio format, e.g., 16:9
        # VEO series models
        elif self.model.startswith("veo"):
            if self.size not in VEO_SUPPORTED_SIZES:
                raise ValueError(
                    f"VEO model only supports resolutions {', '.join(VEO_SUPPORTED_SIZES)}, current is {self.size}"
                )
            if self.seconds not in VEO_SUPPORTED_SECONDS:
                raise ValueError(
                    f"VEO model only supports durations {', '.join(str(s) for s in VEO_SUPPORTED_SECONDS)}, current is {self.seconds}"
                )
        # Sora series models
        elif self.model.startswith("sora"):
            if self.size not in SORA_SUPPORTED_SIZES:
                raise ValueError(
                    f"Sora model only supports resolutions {', '.join(SORA_SUPPORTED_SIZES)}, current is {self.size}"
                )
            if self.seconds not in SORA_SUPPORTED_SECONDS:
                raise ValueError(
                    f"Sora model only supports durations {', '.join(str(s) for s in SORA_SUPPORTED_SECONDS)}, current is {self.seconds}"
                )


    def generate(
        self,
        nodes: List[str],
        final_video_path: str,
        node_output_dir: str,
        characters_text: str,
        scene_text: str,
        station_nodes: List[str],
        time_spans: Optional[List[Tuple[int, int]]] = None,
        style_slug: Optional[str] = None,
        resume_from: int = 1,  # ✅ New: Which node to start generation from
        resume_reference_path: Optional[str] = None,  # ✅ New: Reference image path when resuming generation
    ) -> None:
        if not nodes:
            raise ValueError("Script nodes are empty, cannot generate video.")

        self._validate_params()
        
        # ✅ If resume reference image is provided, use it directly
        if resume_reference_path and os.path.exists(resume_reference_path):
            LOGGER.info(f"Using resume generation mode, reference image: {resume_reference_path}")
            self._reference_spec = ReferenceImageSpec(
                path=resume_reference_path,
                mode=self.reference_mode
            )
        else:
            self._reference_spec = None

        os.makedirs(node_output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(final_video_path), exist_ok=True)

        # ✅ Mechanism 2: Auto-resume - Check if final video already exists
        if os.path.exists(final_video_path):
            file_size = os.path.getsize(final_video_path)
            if file_size > 0:
                LOGGER.info(f"✅ Final video already exists, skipping generation: {final_video_path} ({file_size / 1024 / 1024:.2f} MB)")
                return nodes, station_nodes, False  # ✅ Return original data, no nodes deleted
            else:
                LOGGER.warning(f"Detected empty final video file, will regenerate: {final_video_path}")
                os.remove(final_video_path)

        # ✅ Mechanism 1: Dynamic node list (supports skipping failed nodes)
        active_nodes = list(nodes)
        active_stations = list(station_nodes or [])
        active_time_spans = list(time_spans or [])
        nodes_deleted = False  # ✅ Mark if any nodes were deleted
        
        total = len(active_nodes)
        video_paths: List[str] = []
        
        # ✅ Detect existing video files
        existing_videos, _, _ = detect_existing_nodes(node_output_dir)
        video_paths.extend(existing_videos)
        
        station_sequence = list(active_stations)
        if len(station_sequence) < total:
            filler = station_sequence[-1] if station_sequence else "Station information to be supplemented"
            station_sequence.extend([filler] * (total - len(station_sequence)))

        # ✅ If external reference first frame is provided, first node also uses reference image
        if self.reference_first_frame and os.path.exists(self.reference_first_frame):
            LOGGER.info(f"🎨 Using external reference first frame: {self.reference_first_frame}")
            LOGGER.info("This ensures all models start from the same initial frame")
            # Set reference image for the first node
            if resume_from == 1:
                self._reference_spec = ReferenceImageSpec(
                    path=self.reference_first_frame,
                    mode=self.reference_mode
                )
        
        # ✅ Start generation from specified node
        LOGGER.info(f"Starting generation from node {resume_from} (total {total} nodes)")
        
        current_index = resume_from
        while current_index <= len(active_nodes):
            # Recalculate total (because nodes may have been deleted)
            total = len(active_nodes)
            
            node_text = active_nodes[current_index - 1]
            
            preferred_span: Optional[int] = None
            current_time_span: Optional[Tuple[int, int]] = None
            if active_time_spans and current_index - 1 < len(active_time_spans):
                start_span, end_span = active_time_spans[current_index - 1]
                if end_span > start_span:
                    preferred_span = end_span - start_span
                    current_time_span = (start_span, end_span)
            
            node_seconds = self._determine_node_seconds(node_text, current_index, preferred_span)
            prompt_text = self._build_prompt(
                node_text,
                current_index,
                total,
                characters_text,
                scene_text,
                station_sequence[current_index - 1] if current_index - 1 < len(station_sequence) else "Station information to be supplemented",
                node_seconds,
                current_time_span,
            )
            print("prompt:", prompt_text)

            filename_stub = safe_filename(node_text.split("。")[0])
            if style_slug:
                filename_stub = f"{style_slug}_{current_index:02d}_{filename_stub}"
            else:
                filename_stub = f"{current_index:02d}_{filename_stub}"
            video_path = os.path.join(node_output_dir, f"{filename_stub}.mp4")

            # Prepare reference image parameters
            pic_path_param: Optional[List[str]] = None
            pic_func_param: Optional[List[str]] = None
            if self._reference_spec:
                pic_path_param = [self._reference_spec.path]
                pic_func_param = [self._reference_spec.mode]

            # 🎯 Auto mode: Automatically switch for models that need reference images
            actual_model = self.model
            has_external_first_frame = bool(self.reference_first_frame and current_index == 1)
            
            if self.model == "Wan2.5":
                # If external reference first frame exists, first node also uses i2v
                if current_index == 1 and not has_external_first_frame:
                    actual_model = "wan_t2v"
                    LOGGER.info(f"🔹 Node {current_index}: Using Wan2.5 text-to-video mode (wan_t2v)")
                else:
                    actual_model = "wan_i2v"
                    if has_external_first_frame:
                        LOGGER.info(f"🔹 Node {current_index}: Using external reference first frame + Wan2.5 image-to-video mode (wan_i2v)")
                    else:
                        LOGGER.info(f"🔹 Node {current_index}: Using Wan2.5 image-to-video mode (wan_i2v)")
            elif self.model == "ViduQ2":
                # If external reference first frame exists, first node uses vidu_refer
                if current_index == 1 and not has_external_first_frame:
                    actual_model = "vidu_image"
                    LOGGER.info(f"🔹 Node {current_index}: Using ViduQ2 image-to-video mode (vidu_image)")
                else:
                    actual_model = "vidu_refer"
                    if has_external_first_frame:
                        LOGGER.info(f"🔹 Node {current_index}: Using external reference first frame + ViduQ2 reference-to-video mode (vidu_refer)")
                    else:
                        LOGGER.info(f"🔹 Node {current_index}: Using ViduQ2 reference-to-video mode (vidu_refer)")

            attempt = 0
            error: Optional[str] = None
            while attempt < max(1, self.max_retry):
                attempt += 1
                error = self.api.call_data_eval(
                    text=prompt_text,
                    pic_path=pic_path_param,
                    pic_func=pic_func_param,
                    output_path=video_path,
                    model=actual_model,
                    size=self.size,
                    seconds=node_seconds,
                )
                if error is None:
                    break
                LOGGER.warning("Call failed (attempt %d/%d): %s", attempt, self.max_retry, error)
                # ✅ If it's a content moderation error, stop retrying immediately (to avoid wasting time)
                if is_content_moderation_error(error):
                    LOGGER.info("Detected content moderation error, stopping retry")
                    break
            
            # ✅ Mechanism 1: Error handling - Check if it's an official API error (content sensitive, etc.)
            if error is not None:
                is_api_content_error = is_content_moderation_error(error)
                
                if is_api_content_error:
                    LOGGER.warning(f"⚠️ Node {current_index} triggered content moderation, skipping this node and adjusting script structure")
                    LOGGER.info(f"Deleting node content: {node_text[:50]}...")
                    
                    nodes_deleted = True  # ✅ Mark that nodes were deleted
                    
                    # Delete current node
                    del active_nodes[current_index - 1]
                    
                    # Delete corresponding station node
                    if current_index - 1 < len(station_sequence):
                        del station_sequence[current_index - 1]
                    
                    # Delete corresponding time span
                    if active_time_spans and current_index - 1 < len(active_time_spans):
                        del active_time_spans[current_index - 1]
                    
                    LOGGER.info(f"Remaining nodes: {len(active_nodes)}, continuing generation...")
                    
                    # Don't increment current_index, because after deletion the next node will automatically take the current position
                    continue
                else:
                    # Non-content-sensitive error, raise exception
                    raise RuntimeError(f"Node {current_index} generation failed: {error}")

            self._reference_spec = self._update_reference(video_path, filename_stub, node_output_dir)
            video_paths.append(video_path)
            
            # Successfully generated, move to next node
            current_index += 1

        if not video_paths:
            raise RuntimeError("All nodes failed to generate or were skipped, cannot stitch video")

        LOGGER.info("Node generation completed, starting to stitch %d video segments.", len(video_paths))
        stitch_videos(video_paths, final_video_path)
        LOGGER.info("Stitching completed, final video: %s", final_video_path)
        
        # ✅ Return final node list, station list, and deletion flag
        return active_nodes, station_sequence[:len(active_nodes)], nodes_deleted
    def _select_node_seconds(self, node_text: str, index: int) -> int:
        if not self.tts_estimator:
            return self.seconds

        dialogue_text = extract_dialogue_text(node_text)
        if not dialogue_text:
            LOGGER.debug("Node %d did not detect valid dialogue, using default duration %d seconds.", index, self.seconds)
            return self.seconds

        try:
            estimated = self.tts_estimator.estimate_seconds(dialogue_text)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Node %d speech duration estimation failed: %s, using default duration %d seconds.", index, exc, self.seconds)
            return self.seconds

        padded = estimated + self.duration_padding
        choices = VEO_SUPPORTED_SECONDS if self.model.startswith("veo") else SORA_SUPPORTED_SECONDS
        seconds = self._pick_allowed_duration(padded, choices)

        LOGGER.info(
            "Node %d estimated speech %.2f seconds, after padding %.2f seconds, using duration %d seconds.",
            index,
            estimated,
            padded,
            seconds,
        )
        return seconds

    def _determine_node_seconds(self, node_text: str, index: int, preferred: Optional[int]) -> int:
        base_seconds = self._select_node_seconds(node_text, index)
        if preferred is None:
            return base_seconds

        choices = VEO_SUPPORTED_SECONDS if self.model.startswith("veo") else SORA_SUPPORTED_SECONDS
        allowed = sorted(int(value) for value in choices)
        target = max(4, min(10, preferred))

        candidate = next((value for value in allowed if value >= target and value <= 10), None)
        if candidate is None:
            candidate = next((value for value in reversed(allowed) if value <= 10), None)
        if candidate is None:
            candidate = allowed[0] if allowed else base_seconds

        if base_seconds <= 10 and base_seconds >= candidate:
            return base_seconds
        return candidate

    @staticmethod
    def _pick_allowed_duration(target: float, choices: Sequence[int]) -> int:
        sorted_choices = sorted(choices)
        for value in sorted_choices:
            if target <= value:
                return value
        return sorted_choices[-1] if sorted_choices else int(round(target))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch generate Sora/VEO videos based on script nodes")
    # Test single script parameter settings entry
    parser.add_argument("--script_path", type=str, default="./storyscript.txt", help="Script text file path (UTF-8)", required=False)
    parser.add_argument("--script", type=str, help="Directly passed script text string", required=False)
    # Batch input scripts, batch generate videos

    parser.add_argument("--output_dir", type=str, default="./output_story", help="Output root directory, will create subdirectories by model")
    parser.add_argument(
        "--enable_batch",
        action="store_true",
        help="Enable JSONL batch mode, process response field line by line",
    )
    parser.add_argument("--batch_jsonl", type=str, help="JSONL file path to process in batch mode")
    parser.add_argument(
        "--batch_limit",
        type=int,
        default=0,
        help="Maximum number of scripts in batch mode, 0 means unlimited",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sora2-pro",
        choices=[
            "sora2-pro", "sora2", "veo3.1", "veo3.1-fast",
            "jimeng", "keling",
            "vidu_refer", "vidu_image", "ViduQ2",  # ViduQ2 will auto-switch
            "wan_t2v", "wan_i2v", "Wan2.5"  # Wan2.5 will auto-switch t2v/i2v
        ],
        help="Model name (Wan2.5/ViduQ2 will auto-switch: first node text-only, subsequent nodes use reference images)",
    )

    parser.add_argument(
        "--reference_mode",
        type=str,
        default="first",
        choices=list(REFERENCE_MODE_CHOICES),
        help="Continuity reference mode, options: first/style/asset",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="anime",
        choices=list(STYLE_CHOICES),
        help="Visual style, options: anime/realistic/animated/painterly/abstract/all",
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help="Output video size (if not specified, core models uniformly use 720p: Sora=1280x720, VEO=720p, Wan2.5=720P, ViduQ2=720p, for cross-model reference image sharing)",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=None,
        help="Single segment video duration (if not specified, will auto-select based on model: sora2-pro=12s, veo3.1=8s, Wan2.5=10s, ViduQ2=10s)",
    )
    parser.add_argument("--veo_aspect", type=str, default="16:9", help="VEO aspect ratio, e.g., 16:9")
    parser.add_argument("--veo_audio", action="store_true", help="Whether VEO generates audio")
    parser.add_argument("--app_id", type=str, default="3H9qTcKR_kleinhe", help="Distillation platform model group account")
    parser.add_argument("--app_key", type=str, default="im64DXLqa11LigPq", help="Distillation platform model group secret key")
    parser.add_argument("--log_level", type=str, default="INFO", help="Log level, e.g., INFO/DEBUG")
    parser.add_argument("--max_retry", type=int, default=5, help="Maximum retry count after single segment call failure")
    parser.add_argument(
        "--enable_tts_duration",
        action="store_true",
        help="Enable PaddleSpeech to estimate dialogue speech duration, dynamically set video duration for each node",
    )
    parser.add_argument("--tts_bin", type=str, default="paddlespeech", help="PaddleSpeech executable file path")
    parser.add_argument("--tts_model", type=str, default="fastspeech2_mix", help="PaddleSpeech acoustic model name")
    parser.add_argument("--tts_speaker_id", type=int, default=0, help="PaddleSpeech speaker ID (negative means not specified)")
    parser.add_argument("--tts_sample_rate", type=int, default=24000, help="Synthesized speech sample rate")
    parser.add_argument("--tts_device", type=str, default="cpu", help="PaddleSpeech inference device, e.g., cpu or gpu")
    parser.add_argument("--tts_cache_dir", type=str, default="./tts_duration_cache", help="TTS temporary output directory")
    parser.add_argument("--tts_keep_cache", action="store_true", help="Whether to keep temporarily generated speech files")
    parser.add_argument("--tts_ffprobe_bin", type=str, default="ffprobe", help="ffprobe executable file path")
    parser.add_argument(
        "--tts_duration_padding",
        type=float,
        default=1.5,
        help="Additional padding seconds added on top of estimated speech duration",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Enable resume generation mode, continue from where it was interrupted"
    )
    parser.add_argument(
        "--resume_node_dir",
        type=str,
        help="In resume generation mode, specify the previously generated node folder path"
    )
    parser.add_argument(
        "--reference_first_frame",
        type=str,
        help="Specify external reference first frame image path (for unifying starting frames across different models)"
    )
    return parser


def main() -> None:
    import time
    parser = build_argument_parser()
    args = parser.parse_args()
    start_time = time.time()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s][%(levelname)5s] %(message)s",
    )

    ensure_dependencies()
    
    # Apply model default configuration
    if args.model in MODEL_DEFAULT_CONFIG:
        model_config = MODEL_DEFAULT_CONFIG[args.model]
        # If user hasn't explicitly specified, use model default configuration
        if args.seconds is None:
            args.seconds = model_config["seconds"]
            LOGGER.info(f"Using {args.model} model default duration: {args.seconds} seconds")
        if args.size is None:
            args.size = model_config["size"]
            LOGGER.info(f"Using {args.model} model default resolution: {args.size}")
    else:
        # For models not in MODEL_DEFAULT_CONFIG, use general default values
        if args.seconds is None:
            args.seconds = 12
        if args.size is None:
            args.size = "1792x1024"

    # ✅ Separate parameter checks for batch mode and single script mode
    if args.enable_batch:
        # Batch mode: only needs batch_jsonl, doesn't need script/script_path
        if not args.batch_jsonl:
            raise ValueError("Batch mode requires specifying --batch_jsonl file path.")
        LOGGER.info("Batch mode: Will read scripts from JSONL file")
    else:
        # Single script mode: needs script or script_path
        if not args.script and not args.script_path:
            raise ValueError("Single script mode requires providing either --script or --script_path.")

        script_text = args.script
        if not script_text and args.script_path:
            with open(args.script_path, "r", encoding="utf-8") as file:
                script_text = file.read()

        if not script_text:
            raise ValueError("Read script text is empty.")

        components = extract_story_components(script_text)
        if not components.nodes:
            raise ValueError("Failed to parse valid nodes, please check if script text format is 'number. content'.")
        
        if args.resume and not args.resume_node_dir:
            raise ValueError("When enabling --resume, must specify --resume_node_dir")
        
        if args.resume and args.resume_node_dir:
            if not os.path.exists(args.resume_node_dir):
                raise FileNotFoundError(f"Specified node directory does not exist: {args.resume_node_dir}")
            LOGGER.info(f"Resume generation mode enabled, node directory: {args.resume_node_dir}")
        
        LOGGER.info("Successfully parsed %d nodes.", len(components.nodes))

    api = Api(args.app_id, args.app_key)

    tts_estimator: Optional[PaddlespeechTTSDurationEstimator] = None
    if args.enable_tts_duration:
        speaker_id = args.tts_speaker_id if args.tts_speaker_id >= 0 else None
        try:
            tts_estimator = PaddlespeechTTSDurationEstimator(
                binary=args.tts_bin,
                model_name=args.tts_model,
                sample_rate=args.tts_sample_rate,
                device=args.tts_device,
                temp_dir=args.tts_cache_dir,
                speaker_id=speaker_id,
                ffprobe_bin=args.tts_ffprobe_bin,
                cleanup=not args.tts_keep_cache,
            )
            LOGGER.info("TTS speech duration estimation enabled, model: %s, device: %s", args.tts_model, args.tts_device)
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"TTS duration estimation initialization failed: {exc}") from exc

    style_option = resolve_style_key(args.style)
    style_keys = list(STYLE_ORDER) if style_option == "all" else [style_option]

    generators_by_style: Dict[str, StoryVideoGenerator] = {}
    for style_key in style_keys:
        generators_by_style[style_key] = StoryVideoGenerator(
            api=api,
            output_dir=args.output_dir,
            model=args.model,
            size=args.size,
            seconds=args.seconds,
            max_retry=max(1, args.max_retry),
            reference_mode=args.reference_mode,
            veo_aspect=args.veo_aspect if args.model.startswith("veo") else None,
            veo_audio=bool(args.veo_audio) if args.model.startswith("veo") else None,
            style_key=style_key,
            tts_estimator=tts_estimator,
            duration_padding=args.tts_duration_padding,
            reference_first_frame=args.reference_first_frame,
        )

    style_slugs: Dict[str, str] = {style_key: slugify_model_name(style_key) for style_key in style_keys}
    video_dialogue_entries: Dict[str, str] = {}

    model_dir, node_root, final_root, responses_path, model_slug = prepare_output_structure(
        args.output_dir,
        args.model,
    )
    dialogue_jsonl_path = os.path.join(model_dir, "video_dialogues.jsonl")

    if args.enable_batch:
        LOGGER.info("Batch mode enabled, reading JSONL: %s", args.batch_jsonl)
        LOGGER.info("=" * 60)
        LOGGER.info("📋 Configuration information:")
        LOGGER.info(f"   - Model: {args.model}")
        LOGGER.info(f"   - model_slug: {model_slug}")
        LOGGER.info(f"   - Style: {args.style}")
        LOGGER.info(f"   - style_keys: {style_keys}")
        LOGGER.info(f"   - style_slugs: {style_slugs}")
        LOGGER.info(f"   - Output root directory: {args.output_dir}")
        LOGGER.info(f"   - model_dir: {model_dir}")
        LOGGER.info(f"   - final_root: {final_root}")
        LOGGER.info(f"   - responses_path: {responses_path}")
        LOGGER.info("=" * 60)
        # ✅ Count total
        total_records = sum(1 for _ in iter_batch_responses(args.batch_jsonl))
        LOGGER.info(f"JSONL file has {total_records} valid records")
        
        # ✅ Batch mode: JSONL line N should correspond to index N, not calculated based on responses.jsonl
        # This ensures that even after interruption and rerun, it can correctly skip already generated videos
        start_index = 1
        LOGGER.info(f"Batch mode starting index: {start_index} (JSONL line N → index N)")
        
        # ✅ Scan existing video files (for each style)
        existing_videos_by_style = {}
        for style_key in style_keys:
            existing_indices = scan_existing_videos(
                final_root=final_root,
                model_slug=model_slug,
                style_slug=style_slugs[style_key]
            )
            existing_videos_by_style[style_key] = existing_indices
        
        processed = 0
        for record_index, payload in iter_batch_responses(args.batch_jsonl):
            # ✅ Add debug logs
            LOGGER.info(f"📖 Read line {record_index} record")
            
            if args.batch_limit and processed >= args.batch_limit:
                LOGGER.info("Reached batch processing limit %d, stopping.", args.batch_limit)
                break

            response_text = payload.get("response")
            if not isinstance(response_text, str) or not response_text.strip():
                LOGGER.warning("Line %d missing valid response field, skipping.", record_index)
                continue


            current_index = start_index + processed
            processed += 1  
            
            LOGGER.info("=" * 80)
            LOGGER.info(f"🎬 Processing JSONL line {record_index} → script index #{current_index}")
            LOGGER.info(f"   - JSONL line number: {record_index}")
            LOGGER.info(f"   - Script index: {current_index}")
            LOGGER.info(f"   - Processed count: {processed}")
            
            # ✅ Check if all style videos already exist
            existence_check = {
                style_key: current_index in existing_videos_by_style.get(style_key, set())
                for style_key in style_keys
            }

            
            all_styles_exist = all(existence_check.values())
            
            if all_styles_exist:
                expected_filename = f"{model_slug}_{style_slugs[style_keys[0]]}_{current_index:03d}.mp4"
                LOGGER.info(f"⏭️  Story #{current_index} all styles videos already exist, skip.")

                LOGGER.info("=" * 80)
                continue
            
            LOGGER.info(f"✅ Story #{current_index} needs to be generated, start processing...")
            LOGGER.info("=" * 80)
            
            meta = {k: v for k, v in payload.items() if k != "response"}
            base_metadata: Dict[str, Any] = {"source_line": record_index}
            if meta:
                base_metadata.update(meta)

            try:
                components_batch = extract_story_components(response_text)
                if not components_batch.nodes:
                    raise ValueError(f"Story #{current_index} failed to parse valid nodes.")

                for style_key in style_keys:
                    if current_index in existing_videos_by_style.get(style_key, set()):
                        LOGGER.info(f"⏭️  Story #{current_index} style {style_key} video already exists, skip.")
                        continue
                    components_for_style = StoryComponents(
                        nodes=list(components_batch.nodes),
                        characters=components_batch.characters,
                        scene=components_batch.scene,
                        station_nodes=list(components_batch.station_nodes),
                        dialog_text=components_batch.dialog_text,
                        time_spans=list(components_batch.time_spans),
                    )
                    style_metadata = dict(base_metadata)
                    style_metadata["style"] = style_key
                    style_metadata["character_summary"] = components_for_style.characters
                    style_metadata["scene_summary"] = components_for_style.scene
                    
                    process_story(
                        generators_by_style[style_key],
                        script_text=response_text,
                        model_slug=model_slug,
                        story_index=current_index,
                        node_root=node_root,
                        final_root=final_root,
                        responses_path=responses_path,
                        metadata=style_metadata,
                        components=components_for_style,
                        style_slug=style_slugs[style_key],
                        dialogue_map=video_dialogue_entries,
                        resume_node_dir=args.resume_node_dir if args.resume else None,
                        source_jsonl=args.batch_jsonl, 
                        source_line_no=record_index,  
                    )
                    
                LOGGER.info(f"✅ Story #{current_index} processed successfully.")
                
            except Exception as exc:
                LOGGER.exception(f"❌ Story #{current_index} processing failed: {exc}")
                continue  

        LOGGER.info(f"✅ Batch generation completed, processed {processed} stories.")
        append_video_dialogue_records(dialogue_jsonl_path, video_dialogue_entries)
        return

    # Single script mode
    story_index = determine_next_index(responses_path)
    for style_key in style_keys:
        components_for_style = StoryComponents(
            nodes=list(components.nodes),
            characters=components.characters,
            scene=components.scene,
            station_nodes=list(components.station_nodes),
            dialog_text=components.dialog_text,
            time_spans=list(components.time_spans),
        )
        metadata_single = {
            "mode": "single",
            "style": style_key,
            "character_summary": components_for_style.characters,
            "scene_summary": components_for_style.scene,
        }
        process_story(
            generators_by_style[style_key],
            script_text=script_text,
            model_slug=model_slug,
            story_index=story_index,
            node_root=node_root,
            final_root=final_root,
            responses_path=responses_path,
            metadata=metadata_single,
            components=components_for_style,
            style_slug=style_slugs[style_key],
            dialogue_map=video_dialogue_entries,
            resume_node_dir=args.resume_node_dir if args.resume else None,
            source_jsonl=None, 
            source_line_no=None,
        )

    append_video_dialogue_records(dialogue_jsonl_path, video_dialogue_entries)
    end_time = time.time()
    print(f"Cost time: {end_time - start_time:.2f} s")


if __name__ == "__main__":
    main()
