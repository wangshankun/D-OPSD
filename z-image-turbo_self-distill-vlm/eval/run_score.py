#!/usr/bin/env python
"""Score local evaluation results.

This script keeps the local metrics from the internal scorer and removes the
private image-quality and aesthetic API calls. It computes CLIP image-text
alignment, rewrite-image DINO/LPIPS distance, and VLM concept consistency.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from local_paths import get_local_image_path  # noqa: E402


OBJECT_VLM_SYSTEM_PROMPT = """
You are a strict visual identity evaluator.

You will receive two images:
- The first image is the reference image.
- The second image is the candidate image.

Judge whether the main personalized subject in both images is the same specific
instance, not merely the same category.

Scoring labels:
- same instance: strong evidence that both images show the same specific subject.
- mostly same: probably the same subject, with some generation differences.
- partly similar: some shared visual cues, but likely a different instance.
- different instance: clearly not the same specific subject.

Output exactly one label from this list:
same instance
mostly same
partly similar
different instance
""".strip()


STYLE_VLM_SYSTEM_PROMPT = """
You are a strict visual style consistency evaluator.

You will receive two images:
- The first image is the reference style image.
- The second image is the candidate image.

The subject and composition are allowed to change. Ignore whether the objects,
people, layout, and scene content are the same. Judge only whether the candidate
preserves the reference image's visual style, including color palette, rendering
method, lighting treatment, texture, edge quality, graphic effects, and overall
visual atmosphere.

Scoring labels:
- highly consistent: the candidate strongly preserves the reference style.
- mostly consistent: the candidate preserves most style cues with minor drift.
- slightly consistent: only a few style cues are preserved.
- not consistent: the candidate does not preserve the reference style.

Output exactly one label from this list:
highly consistent
mostly consistent
slightly consistent
not consistent
""".strip()


AUTO_CONCEPT_MODE_SYSTEM_PROMPT = """
You are a strict classifier for DreamBooth/Textual Inversion personalization.

You will receive one reference image plus text metadata/caption for the same
sample. Decide what the special token [V] should represent in downstream
generation and evaluation.

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


OBJECT_LABEL_TO_SCORE = {
    "same instance": 4,
    "mostly same": 3,
    "partly similar": 2,
    "different instance": 1,
}

STYLE_LABEL_TO_SCORE = {
    "highly consistent": 4,
    "mostly consistent": 3,
    "slightly consistent": 2,
    "not consistent": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score local eval JSONL results.")
    parser.add_argument("--jsonl-path", required=True, help="JSONL produced by run_z.py.")
    parser.add_argument(
        "--output-jsonl-path",
        default=None,
        help="Metric-enhanced JSONL path. Defaults to overwriting --jsonl-path.",
    )
    parser.add_argument(
        "--summary-txt-path",
        default=None,
        help="Summary text path. Defaults to <output-jsonl-stem>_summary.txt next to the output JSONL.",
    )
    parser.add_argument(
        "--concept-mode",
        default="auto",
        choices=["auto", "object", "style"],
        help="Concept scoring mode. Auto asks Qwen3-VL to classify source image + caption as object or style.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16", choices=["bfloat16", "float16", "float32"])

    parser.add_argument("--clip-model-name", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2B-s32B-b79K")
    parser.add_argument("--clip-precision", default="fp16")
    parser.add_argument("--skip-clip", action="store_true")

    parser.add_argument("--dinov3-repo", default=None, help="Local DINOv3 repository path.")
    parser.add_argument("--dinov3-weights", default=None, help="Local DINOv3 weight file.")
    parser.add_argument("--dinov3-model-name", default="dinov3_vits16plus")
    parser.add_argument("--skip-dino-lpips", action="store_true")

    parser.add_argument("--qwen-vl-model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--skip-vlm", action="store_true")
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


def save_jsonl(rows: list[dict[str, Any]], jsonl_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_mean(values: list[float | None]) -> float | None:
    clean_values = [v for v in values if v is not None]
    if not clean_values:
        return None
    return float(sum(clean_values) / len(clean_values))


def resolve_path_reference(path_value: str, jsonl_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [
        (jsonl_path.parent / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_rgb_image(path: str | Path) -> Image.Image:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


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


def detect_concept_mode_from_metadata(row: dict[str, Any]) -> str:
    explicit = str(row.get("concept_mode") or row.get("concept_type") or "").strip().lower()
    if explicit in {"object", "style"}:
        return explicit

    joined = "\n".join(str(row.get(key, "")) for key in CONCEPT_MODE_TEXT_KEYS).lower()
    if "[v] style" in joined or "style of [v]" in joined or ("[v]" in joined and "style" in joined):
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

Question: should [V] be evaluated as a concrete subject instance (object) or as
a transferable visual style (style)?

Output exactly one word: object or style.
""".strip()


def parse_concept_mode_label(text: str) -> str | None:
    labels = re.findall(r"\b(object|style)\b", text.strip().lower())
    unique_labels = list(dict.fromkeys(labels))
    if len(unique_labels) == 1:
        return unique_labels[0]
    return None


def judge_auto_concept_mode(
    models: MetricModels,
    source_path: Path,
    row: dict[str, Any],
) -> tuple[str | None, str | None]:
    import torch

    if models.qwen_processor is None or models.qwen_model is None:
        return None, None

    source_image = load_rgb_image(source_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": AUTO_CONCEPT_MODE_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_auto_concept_mode_user_prompt(row)},
                {"type": "image", "image": source_image},
            ],
        },
    ]
    inputs = models.qwen_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(models.device)

    with torch.no_grad():
        generated_ids = models.qwen_model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = models.qwen_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return parse_concept_mode_label(output_text), output_text


def detect_concept_mode(
    row: dict[str, Any],
    configured_mode: str,
    models: MetricModels | None = None,
    source_path: Path | None = None,
    save_raw_output: bool = False,
) -> str:
    if configured_mode != "auto":
        return configured_mode

    if (
        models is not None
        and source_path is not None
        and models.qwen_processor is not None
        and models.qwen_model is not None
    ):
        try:
            concept_mode, raw_output = judge_auto_concept_mode(models, source_path, row)
            if save_raw_output:
                row["auto_concept_mode_vlm_raw_output"] = raw_output
            if concept_mode in {"object", "style"}:
                return concept_mode
            row.setdefault("metric_error", {})["auto_concept_mode_vlm"] = "Could not parse concept mode from VLM output."
        except Exception as exc:
            row.setdefault("metric_error", {})["auto_concept_mode_vlm"] = f"{type(exc).__name__}: {exc}"

    return detect_concept_mode_from_metadata(row)


def replace_identifier_for_score(prompt: str, class_name: str, concept_mode: str) -> str:
    if not isinstance(prompt, str):
        return prompt
    readable_class = class_name.replace("_", " ").strip()
    if concept_mode == "style":
        replacement = f"{readable_class} style" if readable_class else "the learned style"
        prompt = re.sub(r"\[V\]\s*style", replacement, prompt, flags=re.IGNORECASE)
        return re.sub(r"\[V\]", replacement, prompt, flags=re.IGNORECASE)

    replacement = readable_class or "the personalized subject"
    return re.sub(r"\[V\]", replacement, prompt, flags=re.IGNORECASE)


class MetricModels:
    def __init__(self, args: argparse.Namespace):
        import torch

        self.args = args
        self.device = args.device
        self.dtype = dtype_from_name(args.torch_dtype)
        if not str(self.device).startswith("cuda"):
            self.dtype = torch.float32

        self.lpips_model = None
        self.dinov3_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.qwen_processor = None
        self.qwen_model = None

        if not args.skip_dino_lpips:
            self.load_dino_lpips()
        if not args.skip_clip:
            self.load_clip()
        if not args.skip_vlm:
            self.load_qwen_vl()

    def load_dino_lpips(self) -> None:
        import lpips
        import torch

        if not self.args.dinov3_repo or not self.args.dinov3_weights:
            raise ValueError("Pass --dinov3-repo and --dinov3-weights, or use --skip-dino-lpips.")

        print("Loading LPIPS...")
        self.lpips_model = lpips.LPIPS(net="vgg").to(device=self.device)
        self.lpips_model.eval()

        print("Loading DINOv3...")
        self.dinov3_model = torch.hub.load(
            self.args.dinov3_repo,
            self.args.dinov3_model_name,
            source="local",
            weights=self.args.dinov3_weights,
        ).to(device=self.device, dtype=self.dtype)
        self.dinov3_model.eval()

    def load_clip(self) -> None:
        from open_clip import create_model_and_transforms, get_tokenizer

        print("Loading CLIP...")
        self.clip_model, _, self.clip_preprocess = create_model_and_transforms(
            self.args.clip_model_name,
            pretrained=self.args.clip_pretrained,
            precision=self.args.clip_precision,
            device=self.device,
        )
        self.clip_model.eval()
        self.clip_tokenizer = get_tokenizer(self.args.clip_model_name)

    def load_qwen_vl(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        print("Loading Qwen3-VL...")
        self.qwen_processor = AutoProcessor.from_pretrained(self.args.qwen_vl_model_path)
        model_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.args.qwen_vl_model_path,
            "torch_dtype": torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32,
        }
        if self.args.attn_implementation:
            model_kwargs["attn_implementation"] = self.args.attn_implementation
        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(**model_kwargs).to(self.device)
        self.qwen_model.eval()


def process_one_pair_dino_lpips(models: MetricModels, source_path: Path, generated_path: Path) -> dict[str, float]:
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF

    if models.dinov3_model is None or models.lpips_model is None:
        return {"dino_distance": None, "lpips_distance": None}

    with torch.no_grad():
        source_image = load_rgb_image(source_path)
        generated_image = load_rgb_image(generated_path)
        source_image = source_image.resize(generated_image.size, Image.Resampling.BICUBIC)

        source_tensor = (
            torch.from_numpy(np.array(source_image))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=models.device, dtype=models.dtype)
            / 255.0
        )
        generated_tensor = (
            torch.from_numpy(np.array(generated_image))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=models.device, dtype=models.dtype)
            / 255.0
        )

        source_dino_input = TF.normalize(
            source_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        generated_dino_input = TF.normalize(
            generated_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        source_features = models.dinov3_model.forward_features(source_dino_input)["x_norm_patchtokens"]
        generated_features = models.dinov3_model.forward_features(generated_dino_input)["x_norm_patchtokens"]
        source_global = F.normalize(source_features.mean(dim=1).float(), dim=-1)
        generated_global = F.normalize(generated_features.mean(dim=1).float(), dim=-1)
        dino_similarity = (source_global * generated_global).sum(dim=-1).item()

        lpips_source = (source_tensor.float() * 2.0) - 1.0
        lpips_generated = (generated_tensor.float() * 2.0) - 1.0
        lpips_distance = models.lpips_model(lpips_source, lpips_generated).reshape(-1).item()

    return {
        "dino_distance": float(1.0 - dino_similarity),
        "lpips_distance": float(lpips_distance),
    }


def score_image_text_pair_clip(models: MetricModels, image_path: Path, prompt_en: str) -> dict[str, float | None]:
    import torch

    if models.clip_model is None or models.clip_preprocess is None or models.clip_tokenizer is None:
        return {"clip_score": None}

    image = load_rgb_image(image_path)
    image_input = models.clip_preprocess(image).unsqueeze(0).to(models.device)
    text_tokens = models.clip_tokenizer([prompt_en]).to(models.device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(models.device).startswith("cuda")):
        image_features = models.clip_model.encode_image(image_input)
        text_features = models.clip_model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        clip_score = torch.diagonal(image_features @ text_features.t()).mean().item()

    return {"clip_score": float(clip_score)}


def parse_vlm_label(text: str, concept_mode: str) -> tuple[str | None, int | None]:
    lowered = text.strip().lower()
    mapping = STYLE_LABEL_TO_SCORE if concept_mode == "style" else OBJECT_LABEL_TO_SCORE
    for label, score in mapping.items():
        if label in lowered:
            return label, score
    return None, None


def judge_concept_consistency(
    models: MetricModels,
    reference_path: Path,
    candidate_path: Path,
    concept_mode: str,
) -> dict[str, Any]:
    import torch

    if models.qwen_processor is None or models.qwen_model is None:
        return {
            "vlm_concept_consistency_label": None,
            "vlm_concept_consistency_score": None,
            "vlm_raw_output": None,
        }

    reference_image = load_rgb_image(reference_path)
    candidate_image = load_rgb_image(candidate_path)
    if concept_mode == "style":
        system_prompt = STYLE_VLM_SYSTEM_PROMPT
        user_prompt = "Judge whether the second image preserves the visual style of the first image. Output one label only."
    else:
        system_prompt = OBJECT_VLM_SYSTEM_PROMPT
        user_prompt = "Judge whether the two images show the same specific personalized subject. Output one label only."

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image", "image": reference_image},
                {"type": "image", "image": candidate_image},
            ],
        },
    ]
    inputs = models.qwen_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(models.device)

    with torch.no_grad():
        generated_ids = models.qwen_model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = models.qwen_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    label, score = parse_vlm_label(output_text, concept_mode)

    return {
        "vlm_concept_consistency_label": label,
        "vlm_concept_consistency_score": score,
        "vlm_raw_output": output_text,
    }


def ensure_metric_fields(row: dict[str, Any]) -> None:
    row.setdefault("rewrite_metrics", {})
    row.setdefault("test_metrics", [])
    row.setdefault("summary_metrics", {})
    row.setdefault("metric_error", {})


def get_prompt_fields_for_score(row: dict[str, Any], concept_mode: str) -> tuple[str, list[str]]:
    class_name = str(row.get("class_name", "")).strip()
    rewrite = str(row.get("rewrite_prompt_en") or row.get("rewrite_prompt_zh") or "")
    rewrite = replace_identifier_for_score(rewrite, class_name, concept_mode)

    test_prompts_raw = row.get("test_prompts_en")
    if not isinstance(test_prompts_raw, list):
        test_prompts_raw = row.get("test_prompts_zh", [])
    test_prompts = [
        replace_identifier_for_score(prompt, class_name, concept_mode)
        for prompt in test_prompts_raw
        if isinstance(prompt, str)
    ]
    return rewrite, test_prompts[:4]


def get_generated_paths(row: dict[str, Any]) -> tuple[str, list[str]]:
    rewrite_path = str(row.get("rewrite_prompt_en_path") or row.get("rewrite_prompt_zh_path") or "")
    test_paths_raw = row.get("test_prompts_en_paths")
    if not isinstance(test_paths_raw, list):
        test_paths_raw = row.get("test_prompts_zh_paths", [])
    test_paths = [str(path) for path in test_paths_raw[:4]]
    return rewrite_path, test_paths


def evaluate_one_sample(row: dict[str, Any], jsonl_path: Path, models: MetricModels, configured_mode: str) -> dict[str, Any]:
    ensure_metric_fields(row)

    try:
        source_path = Path(get_local_image_path(row, jsonl_path=jsonl_path, data_root=PROJECT_ROOT))
    except Exception as exc:
        row["metric_error"]["source_image"] = f"{type(exc).__name__}: {exc}"
        source_path = None

    concept_mode = detect_concept_mode(
        row,
        configured_mode,
        models=models,
        source_path=source_path,
        save_raw_output=getattr(models.args, "save_auto_concept_mode_raw_output", False),
    )
    row["concept_mode"] = concept_mode

    rewrite_path_raw, test_paths_raw = get_generated_paths(row)
    rewrite_prompt_for_score, test_prompts_for_score = get_prompt_fields_for_score(row, concept_mode)
    row["rewrite_prompt_en_for_score"] = rewrite_prompt_for_score
    row["test_prompts_en_for_score"] = test_prompts_for_score

    rewrite_path = resolve_path_reference(rewrite_path_raw, jsonl_path) if rewrite_path_raw else None
    if source_path and rewrite_path:
        try:
            row["rewrite_metrics"].update(process_one_pair_dino_lpips(models, source_path, rewrite_path))
        except Exception as exc:
            row["rewrite_metrics"]["dino_distance"] = None
            row["rewrite_metrics"]["lpips_distance"] = None
            row["metric_error"]["rewrite_dino_lpips"] = f"{type(exc).__name__}: {exc}"
    else:
        row["rewrite_metrics"].setdefault("dino_distance", None)
        row["rewrite_metrics"].setdefault("lpips_distance", None)

    if rewrite_path and rewrite_prompt_for_score.strip():
        try:
            row["rewrite_metrics"].update(score_image_text_pair_clip(models, rewrite_path, rewrite_prompt_for_score))
            row["rewrite_metrics"]["clip_prompt_en"] = rewrite_prompt_for_score
        except Exception as exc:
            row["rewrite_metrics"]["clip_score"] = None
            row["rewrite_metrics"]["clip_prompt_en"] = rewrite_prompt_for_score
            row["metric_error"]["rewrite_clip_score"] = f"{type(exc).__name__}: {exc}"
    else:
        row["rewrite_metrics"].setdefault("clip_score", None)
        row["rewrite_metrics"].setdefault("clip_prompt_en", rewrite_prompt_for_score)

    n = min(4, max(len(test_paths_raw), len(test_prompts_for_score)))
    while len(row["test_metrics"]) < n:
        row["test_metrics"].append({})

    for index in range(n):
        item_metric = row["test_metrics"][index]
        test_path_raw = test_paths_raw[index] if index < len(test_paths_raw) else ""
        test_prompt = test_prompts_for_score[index] if index < len(test_prompts_for_score) else ""
        test_path = resolve_path_reference(test_path_raw, jsonl_path) if test_path_raw else None

        if test_path and test_prompt.strip():
            try:
                item_metric.update(score_image_text_pair_clip(models, test_path, test_prompt))
                item_metric["clip_prompt_en"] = test_prompt
            except Exception as exc:
                item_metric["clip_score"] = None
                item_metric["clip_prompt_en"] = test_prompt
                row["metric_error"][f"test_{index}_clip_score"] = f"{type(exc).__name__}: {exc}"
        else:
            item_metric.setdefault("clip_score", None)
            item_metric.setdefault("clip_prompt_en", test_prompt)

        if source_path and test_path:
            try:
                item_metric.update(judge_concept_consistency(models, source_path, test_path, concept_mode))
            except Exception as exc:
                item_metric["vlm_concept_consistency_label"] = None
                item_metric["vlm_concept_consistency_score"] = None
                item_metric["vlm_raw_output"] = None
                row["metric_error"][f"test_{index}_vlm"] = f"{type(exc).__name__}: {exc}"
        else:
            item_metric.setdefault("vlm_concept_consistency_label", None)
            item_metric.setdefault("vlm_concept_consistency_score", None)
            item_metric.setdefault("vlm_raw_output", None)

    merged_clip = [row["rewrite_metrics"].get("clip_score")] + [
        item.get("clip_score") for item in row["test_metrics"]
    ]
    merged_vlm = [item.get("vlm_concept_consistency_score") for item in row["test_metrics"]]

    row["summary_metrics"]["mean_clip_score"] = safe_mean(merged_clip)
    row["summary_metrics"]["mean_vlm_concept_consistency_score"] = safe_mean(merged_vlm)
    row["summary_metrics"]["rewrite_dino_distance"] = row["rewrite_metrics"].get("dino_distance")
    row["summary_metrics"]["rewrite_lpips_distance"] = row["rewrite_metrics"].get("lpips_distance")
    return row


def summarize_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    merged_clip_score = []
    merged_vlm_score = []
    rewrite_dino_distance = []
    rewrite_lpips_distance = []

    for row in rows:
        rewrite_metrics = row.get("rewrite_metrics", {})
        test_metrics = row.get("test_metrics", [])

        if rewrite_metrics.get("clip_score") is not None:
            merged_clip_score.append(rewrite_metrics["clip_score"])
        if rewrite_metrics.get("dino_distance") is not None:
            rewrite_dino_distance.append(rewrite_metrics["dino_distance"])
        if rewrite_metrics.get("lpips_distance") is not None:
            rewrite_lpips_distance.append(rewrite_metrics["lpips_distance"])

        for item in test_metrics:
            if item.get("clip_score") is not None:
                merged_clip_score.append(item["clip_score"])
            if item.get("vlm_concept_consistency_score") is not None:
                merged_vlm_score.append(item["vlm_concept_consistency_score"])

    return {
        "mean_clip_score": safe_mean(merged_clip_score),
        "mean_vlm_concept_consistency_score": safe_mean(merged_vlm_score),
        "rewrite_dino_distance": safe_mean(rewrite_dino_distance),
        "rewrite_lpips_distance": safe_mean(rewrite_lpips_distance),
        "count_clip_score": len(merged_clip_score),
        "count_vlm_concept_consistency_score": len(merged_vlm_score),
        "count_rewrite_dino_distance": len(rewrite_dino_distance),
        "count_rewrite_lpips_distance": len(rewrite_lpips_distance),
    }


def save_summary_txt(summary: dict[str, Any], summary_txt_path: Path) -> None:
    summary_txt_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_keys = [
        "mean_clip_score",
        "mean_vlm_concept_consistency_score",
        "rewrite_dino_distance",
        "rewrite_lpips_distance",
        "count_clip_score",
        "count_vlm_concept_consistency_score",
        "count_rewrite_dino_distance",
        "count_rewrite_lpips_distance",
    ]
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        for key in ordered_keys:
            value = summary.get(key)
            line = f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}"
            print(line)
            f.write(line + "\n")
    print(f"Saved summary to {summary_txt_path}")


def main() -> None:
    args = parse_args()

    from tqdm import tqdm

    jsonl_path = Path(args.jsonl_path).expanduser().resolve()
    output_jsonl_path = (
        Path(args.output_jsonl_path).expanduser().resolve()
        if args.output_jsonl_path
        else jsonl_path
    )
    summary_txt_path = (
        Path(args.summary_txt_path).expanduser().resolve()
        if args.summary_txt_path
        else output_jsonl_path.with_name(f"{output_jsonl_path.stem}_summary.txt")
    )

    print(f"Loading rows from {jsonl_path}")
    rows = load_jsonl(jsonl_path)
    print(f"Loaded {len(rows)} rows")

    models = MetricModels(args)
    scored_rows = []
    for row in tqdm(rows, desc="Scoring"):
        try:
            scored_rows.append(evaluate_one_sample(row, jsonl_path, models, args.concept_mode))
        except Exception as exc:
            row.setdefault("metric_error", {})
            row["metric_error"]["sample_level_fatal"] = f"{type(exc).__name__}: {exc}"
            scored_rows.append(row)

    save_jsonl(scored_rows, output_jsonl_path)
    summary = summarize_dataset(scored_rows)
    save_summary_txt(summary, summary_txt_path)
    print(f"Saved scored JSONL to {output_jsonl_path}")


if __name__ == "__main__":
    main()
