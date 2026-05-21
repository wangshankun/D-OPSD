#!/usr/bin/env python
"""Generate local evaluation prompts with Qwen3-VL.

The script reads one or more local JSONL datasets, asks a VLM to rewrite the
source prompt, and creates four held-out test prompts per row. It supports both
object personalization and style personalization. The placeholder token is
always ``[V]``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_paths import get_local_image_path, resolve_existing_path  # noqa: E402


REWRITE_SYSTEM_PROMPT = """
# Role
You are an expert image understanding and text-to-image evaluation assistant.

# Task
Given one image and its existing English descriptions, produce:
1. One rewritten English prompt that preserves the original intent.
2. Four new English test prompts for evaluating personalization generalization.

# Shared Rules
1. Use the placeholder token exactly as [V].
2. Never use [v*], <v*>, [V*], or any other placeholder variant.
3. Output valid JSON only. Do not include markdown or explanations.
4. The prompts must be natural, concrete, visual, and directly usable for text-to-image generation.
5. The four test prompts must be clearly different from each other.

# Object Personalization Rules
Use these rules when the concept mode is "object":
1. [V] is the personalized subject.
2. Any place that refers to the personalized subject must use only [V].
3. Do not describe [V]'s category, identity, color, material, texture, age, gender, or structure.
4. The rewritten prompt must keep the original scene intent while changing wording and sentence order.
5. The four test prompts must keep [V] as the subject and change the scene, background, lighting,
   weather, camera view, or composition.
6. At least two test prompts should be realistic everyday scenes, and at least two should be
   imaginative, surreal, or fantasy scenes.

# Style Personalization Rules
Use these rules when the concept mode is "style":
1. [V] is the personalized visual style.
2. Every prompt must preserve the [V] style.
3. The rewritten prompt must keep the original intent while changing wording and sentence order.
4. The four test prompts must change the main subject and composition while preserving the [V] style.
5. The four test prompts should cover meaningfully different subjects, layouts, environments,
   lighting conditions, and camera views.

# Output Format
{
  "rewrite_prompt_en": "...",
  "test_prompts_en": [
    "...",
    "...",
    "...",
    "..."
  ]
}
""".strip()


AUTO_CONCEPT_MODE_SYSTEM_PROMPT = """
You are a strict classifier for DreamBooth/Textual Inversion personalization.

You will receive one reference image plus text metadata/caption for the same
sample. Decide what the special token [V] should represent in downstream
prompt generation.

Labels:
- object: [V] denotes one concrete, repeatable subject instance, such as a
  particular person, animal, toy, product, object, character, or logo. Choose
  object when the caption/image center on the identity of a specific tangible
  subject, even if words like photo, cartoon, 3D render, watercolor, anime, or
  cinematic appear as ordinary visual descriptors.
- style: [V] denotes a transferable visual style, medium, rendering method,
  artistic treatment, aesthetic, texture language, or overall visual atmosphere.
  Choose style when future prompts should change the subject/content while
  preserving the visual treatment, especially when the text says "[V] style",
  "style of [V]", "in [V]", or similar.

Decision rules:
1. Use the text metadata first to infer what [V] names, then use the image as
   visual evidence.
2. Do not classify as style merely because the image has a strong appearance or
   the caption contains style adjectives. If [V] names a specific subject, the
   answer is object.
3. Do not classify as object merely because the image contains objects. If [V]
   names the visual treatment applied to many possible subjects, the answer is
   style.
4. If both a subject and a style are present, choose the role that [V] plays in
   the caption/metadata.
5. If uncertain, choose object only when a concrete subject identity is clearly
   learnable from the image and text; otherwise choose style.

Output exactly one lowercase word, either "object" or "style". Do not output
both words and do not add any explanation.
""".strip()


CONCEPT_MODE_TEXT_KEYS = (
    "class_name",
    "short",
    "medium",
    "detailed",
    "user_prompt",
    "prompt",
    "caption",
    "short_en",
    "medium_en",
    "detailed_en",
    "user_prompt_en",
    "prompt_en",
    "caption_en",
    "short_zh",
    "medium_zh",
    "detailed_zh",
    "user_prompt_zh",
    "prompt_zh",
    "caption_zh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local eval prompts with Qwen3-VL.")
    parser.add_argument("--jsonl-paths", nargs="+", required=True, help="Input local JSONL files.")
    parser.add_argument("--output-jsonl-path", required=True, help="Merged output JSONL path.")
    parser.add_argument("--qwen-vl-model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional attention implementation, for example flash_attention_2 or eager.",
    )
    parser.add_argument(
        "--concept-mode",
        default="auto",
        choices=["auto", "object", "style"],
        help="Prompt generation mode. Auto asks Qwen3-VL to classify source image + caption as object or style.",
    )
    parser.add_argument("--max-items-per-jsonl", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--min-image-side", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output JSONL if it exists.")
    parser.add_argument(
        "--save-auto-concept-mode-raw-output",
        action="store_true",
        help="Save the raw Qwen3-VL output used by --concept-mode auto for debugging.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_jsonl(jsonl_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(row: dict[str, Any], output_jsonl_path: Path) -> None:
    with open(output_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_from_response(text: str) -> str | None:
    text = text.strip()

    for candidate in (
        text,
        re.sub(r"^```json\s*", "", re.sub(r"\s*```$", "", text)),
    ):
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None

    candidate = match.group(0)
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        return None


def resize_if_needed(image: Image.Image, min_side: int) -> Image.Image:
    from PIL import Image

    width, height = image.size
    if width >= min_side or height >= min_side:
        return image

    if width > height:
        new_width = min_side
        new_height = int(min_side * height / width)
    else:
        new_height = min_side
        new_width = int(min_side * width / height)
    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)


def read_image(row: dict[str, Any], jsonl_path: Path, min_image_side: int) -> Image.Image:
    from PIL import Image

    image_path = get_local_image_path(row, jsonl_path=jsonl_path, data_root=PROJECT_ROOT)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    return resize_if_needed(image, min_image_side)


def detect_concept_mode_from_metadata(row: dict[str, Any]) -> str:
    explicit = str(row.get("concept_mode") or row.get("concept_type") or "").strip().lower()
    if explicit in {"object", "style"}:
        return explicit

    joined = "\n".join(str(row.get(key, "")) for key in CONCEPT_MODE_TEXT_KEYS).lower()
    if "[v] style" in joined or "style of [v]" in joined or ("in [v]" in joined and "style" in joined):
        return "style"
    return "object"


def build_auto_concept_mode_user_prompt(row: dict[str, Any]) -> str:
    lines = []
    for key in CONCEPT_MODE_TEXT_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value.strip()}")

    metadata = "\n".join(lines) if lines else "(no text metadata)"
    if len(metadata) > 5000:
        metadata = metadata[:5000] + "\n...(truncated)"

    return f"""
Classify this personalization sample.

Text metadata/caption:
{metadata}

Question: should [V] be used as a concrete subject instance (object) or as a
transferable visual style (style) when generating evaluation prompts?

Output exactly one word: object or style.
""".strip()


def parse_concept_mode_label(text: str) -> str | None:
    labels = re.findall(r"\b(object|style)\b", text.strip().lower())
    unique_labels = list(dict.fromkeys(labels))
    if len(unique_labels) == 1:
        return unique_labels[0]
    return None


def judge_auto_concept_mode(
    image: Image.Image,
    row: dict[str, Any],
    processor: AutoProcessor,
    model: Qwen3VLForConditionalGeneration,
    device: str,
) -> tuple[str | None, str]:
    import torch

    messages = [
        {"role": "system", "content": [{"type": "text", "text": AUTO_CONCEPT_MODE_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_auto_concept_mode_user_prompt(row)},
                {"type": "image", "image": image},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return parse_concept_mode_label(output_text), output_text


def detect_concept_mode(
    row: dict[str, Any],
    configured_mode: str,
    image: Image.Image | None = None,
    processor: AutoProcessor | None = None,
    model: Qwen3VLForConditionalGeneration | None = None,
    device: str = "cuda",
) -> tuple[str, str | None]:
    if configured_mode != "auto":
        return configured_mode, None

    if image is not None and processor is not None and model is not None:
        try:
            concept_mode, raw_output = judge_auto_concept_mode(image, row, processor, model, device)
            if concept_mode in {"object", "style"}:
                return concept_mode, raw_output
            return detect_concept_mode_from_metadata(row), raw_output
        except Exception as exc:
            return detect_concept_mode_from_metadata(row), f"{type(exc).__name__}: {exc}"

    return detect_concept_mode_from_metadata(row), None


def build_user_prompt(row: dict[str, Any], concept_mode: str) -> str:
    fields = {
        "prompt": row.get("prompt", ""),
        "caption": row.get("caption", ""),
        "short_en": row.get("short_en", ""),
        "medium_en": row.get("medium_en", ""),
        "detailed_en": row.get("detailed_en", ""),
        "user_prompt_en": row.get("user_prompt_en", ""),
        "prompt_en": row.get("prompt_en", ""),
        "caption_en": row.get("caption_en", ""),
    }
    class_name = str(row.get("class_name", "")).strip()

    if concept_mode == "style":
        mode_instruction = (
            "Concept mode: style. [V] denotes the target visual style. "
            "Create test prompts that change the subject and composition while preserving [V] style."
        )
    else:
        mode_instruction = (
            "Concept mode: object. [V] denotes the target subject. "
            "Create test prompts that keep [V] as the only subject identifier and change the scene."
        )

    return f"""
Use the image and the existing English descriptions to complete the task.

{mode_instruction}

Known metadata:
class_name: {class_name}
prompt: {fields["prompt"]}
caption: {fields["caption"]}
short_en: {fields["short_en"]}
medium_en: {fields["medium_en"]}
detailed_en: {fields["detailed_en"]}
user_prompt_en: {fields["user_prompt_en"]}
prompt_en: {fields["prompt_en"]}
caption_en: {fields["caption_en"]}

Requirements:
1. Produce one rewritten English prompt that preserves the source intent.
2. Produce four new English test prompts.
3. Every output prompt must contain [V].
4. Do not use [v*] or any placeholder other than [V].
5. Return valid JSON with keys rewrite_prompt_en and test_prompts_en.
""".strip()


def generate_rewrite_and_tests(
    image: Image.Image,
    row: dict[str, Any],
    concept_mode: str,
    processor: AutoProcessor,
    model: Qwen3VLForConditionalGeneration,
    device: str,
    max_new_tokens: int,
) -> tuple[dict[str, Any] | None, str]:
    import torch

    user_prompt = build_user_prompt(row, concept_mode)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": REWRITE_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": image},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    output_json = extract_json_from_response(output_text)
    if output_json is None:
        return None, output_text

    try:
        return json.loads(output_json), output_text
    except Exception:
        return None, output_text


def validate_output(generated: dict[str, Any], concept_mode: str, class_name: str) -> bool:
    rewrite = generated.get("rewrite_prompt_en", "")
    tests = generated.get("test_prompts_en", [])

    if not isinstance(rewrite, str) or "[V]" not in rewrite or "[v*]" in rewrite:
        return False
    if not isinstance(tests, list) or len(tests) < 4:
        return False

    banned_words: list[str] = []
    if concept_mode == "object" and class_name:
        banned_words.append(class_name.lower())

    for text in [rewrite] + tests[:4]:
        if not isinstance(text, str):
            return False
        if "[V]" not in text or "[v*]" in text:
            return False
        lowered = text.lower()
        if any(word and word in lowered for word in banned_words):
            return False

    return True


def main() -> None:
    args = parse_args()

    from tqdm import tqdm
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    output_jsonl_path = Path(args.output_jsonl_path).expanduser().resolve()
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists. Pass --overwrite to replace it: {output_jsonl_path}")
        output_jsonl_path.unlink()

    dtype = dtype_from_name(args.torch_dtype)
    processor = AutoProcessor.from_pretrained(args.qwen_vl_model_path)
    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": args.qwen_vl_model_path,
        "torch_dtype": dtype,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen3VLForConditionalGeneration.from_pretrained(**model_kwargs).to(args.device)
    model.eval()

    total_written = 0
    for jsonl_path_arg in args.jsonl_paths:
        jsonl_path = resolve_existing_path(jsonl_path_arg, PROJECT_ROOT)
        rows = load_jsonl(jsonl_path)
        if args.max_items_per_jsonl > 0:
            rows = rows[: args.max_items_per_jsonl]

        print(f"Processing {jsonl_path} ({len(rows)} rows)")
        for row_index, row in enumerate(tqdm(rows, desc=jsonl_path.name)):
            try:
                image = read_image(row, jsonl_path, args.min_image_side)
                concept_mode, concept_mode_raw_output = detect_concept_mode(
                    row,
                    args.concept_mode,
                    image=image,
                    processor=processor,
                    model=model,
                    device=args.device,
                )
                generated, raw_output = generate_rewrite_and_tests(
                    image=image,
                    row=row,
                    concept_mode=concept_mode,
                    processor=processor,
                    model=model,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                )
                if generated is None:
                    print(f"[WARN] Could not parse JSON for row {row_index}: {raw_output}")
                    continue
                if not validate_output(generated, concept_mode, str(row.get("class_name", ""))):
                    print(f"[WARN] Invalid VLM output for row {row_index}: {raw_output}")
                    continue

                out_row = dict(row)
                out_row["concept_mode"] = concept_mode
                if args.save_auto_concept_mode_raw_output and concept_mode_raw_output is not None:
                    out_row["auto_concept_mode_vlm_raw_output"] = concept_mode_raw_output
                out_row["rewrite_prompt_en"] = generated["rewrite_prompt_en"]
                out_row["test_prompts_en"] = generated["test_prompts_en"][:4]
                append_jsonl(out_row, output_jsonl_path)
                total_written += 1
            except Exception as exc:
                print(f"[ERROR] {jsonl_path}, row {row_index}: {type(exc).__name__}: {exc}")

    print(f"Done. Wrote {total_written} rows to {output_jsonl_path}")


if __name__ == "__main__":
    main()
