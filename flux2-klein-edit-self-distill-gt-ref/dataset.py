import json
import random
import torch
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
from typing import List, Tuple, Dict, Hashable
from torchvision.transforms import functional as F
from local_paths import resolve_existing_path, resolve_image_path

# --- Utility functions ---
def parse_ratios(ratio_strs: List[str]) -> List[Tuple[int, int]]:
    ratios = []
    for s in ratio_strs:
        res = s.split(' ')[0]
        w, h = map(int, res.split('x'))
        ratios.append((w, h))
    return ratios

def has_transparency(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA"):
        return True
    if img.mode == "P" and "transparency" in img.info:
        return True
    return False

def to_rgb_safely(img: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    if has_transparency(img):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, bg)
        background.paste(img, mask=img.getchannel("A"))
        return background
    return img.convert("RGB")


IMAGE_LIST_KEYS = (
    "local_path_list",
    "local_paths",
    "image_path_list",
    "image_paths",
)

PROMPT_FALLBACK_KEYS = (
    "user_prompt_en",
    "short_en",
    "detailed_en",
    "medium_en",
    "user_prompt_zh",
    "short_zh",
    "detailed_zh",
    "medium_zh",
)


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def get_edit_image_paths(item, jsonl_path: str | Path | None = None, data_root: str | Path | None = None) -> List[str]:
    for key in IMAGE_LIST_KEYS:
        if key in item:
            paths = _as_list(item[key])
            if len(paths) < 2:
                raise ValueError("edit samples need at least one reference image and one target image")
            return [str(resolve_image_path(str(path), jsonl_path=jsonl_path, data_root=data_root)) for path in paths]
    raise KeyError(f"item must contain one of {IMAGE_LIST_KEYS}")


def get_prompt_text(item, prompt_key: str) -> Tuple[str, str]:
    keys = [prompt_key] + [key for key in PROMPT_FALLBACK_KEYS if key != prompt_key]
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value), key
    return "", prompt_key


def _parse_hw_value(value, key: str) -> Tuple[float, float]:
    first, second = str(value).split("*")
    if key == "h*w":
        return float(first), float(second)
    return float(second), float(first)


# --- 1. Dataset ---
class TextImageDataset(Dataset):
    def __init__(self, jsonl_path: str, target_resolutions: List[Tuple[int, int]], data_root: str | Path | None = None):
        self.data = []
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else Path(__file__).resolve().parent
        self.jsonl_path = resolve_existing_path(jsonl_path, self.data_root)
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.target_resolutions = target_resolutions
        self.buckets: Dict[Hashable, List[int]] = {}
        self.index_bucket_keys: Dict[int, Hashable] = {}
        self._build_buckets()

    def _best_bucket_idx(self, width: float, height: float) -> int:
        orig_ratio = width / height
        best_ratio_idx = 0
        min_diff = float('inf')
        for i, (tw, th) in enumerate(self.target_resolutions):
            target_ratio = tw / th
            diff = abs(math.log(orig_ratio) - math.log(target_ratio))
            if diff < min_diff:
                min_diff = diff
                best_ratio_idx = i
        return best_ratio_idx

    def _item_sizes(self, item) -> List[Tuple[float, float]]:
        for key in ("h*w", "w*h"):
            if key in item:
                values = _as_list(item[key])
                return [_parse_hw_value(value, key) for value in values]

        image_paths = get_edit_image_paths(item, jsonl_path=self.jsonl_path, data_root=self.data_root)
        sizes = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                w, h = img.size
            sizes.append((float(h), float(w)))
        return sizes

    def _build_buckets(self):
        for idx, item in enumerate(self.data):
            sizes = self._item_sizes(item)
            bucket_indices = [self._best_bucket_idx(width=w, height=h) for h, w in sizes]
            target_bucket_idx = bucket_indices[-1]
            bucket_key = (target_bucket_idx, len(sizes) - 1)
            self.buckets.setdefault(bucket_key, []).append(idx)
            self.index_bucket_keys[idx] = bucket_key

    def process_image(self, image: Image.Image, target_size: Tuple[int, int]):
        tw, th = target_size
        w, h = image.size
        scale = max(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        image = image.crop((left, top, left + tw, top + th))
        img_tensor = F.to_tensor(image)
        img_tensor = F.normalize(img_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return img_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, info):
        # Normalize the input info format
        if isinstance(info, (tuple, list)):
            if len(info) == 3:
                idx, target_res, prompt_key = info
                retry_count = 0
            elif len(info) == 4:
                idx, target_res, prompt_key, retry_count = info
            else:
                raise ValueError(f"Invalid info length: {len(info)}")
        else:
            # Handle corner cases where only an integer index is passed
            idx = info
            target_res = self.target_resolutions[0]
            prompt_key = "detailed_en"  # default value
            retry_count = 0

        if retry_count > 10:
            raise RuntimeError(f"Retry limit reached for index {idx}")

        item = self.data[idx]
        try:
            image_paths = get_edit_image_paths(item, jsonl_path=self.jsonl_path, data_root=self.data_root)
            reference_pixel_values = []
            for image_path in image_paths[:-1]:
                with Image.open(image_path) as img:
                    image = to_rgb_safely(img)
                reference_pixel_values.append(self.process_image(image, target_res))

            with Image.open(image_paths[-1]) as img:
                target_image = to_rgb_safely(img)
            target_pixel_values = self.process_image(target_image, target_res)
            reference_pixel_values = torch.stack(reference_pixel_values, dim=0)
        except Exception as e:
            bucket_key = self.index_bucket_keys.get(idx, (self.target_resolutions.index(target_res), 1))
            new_idx = random.choice(self.buckets[bucket_key])
            return self.__getitem__((new_idx, target_res, prompt_key, retry_count + 1))

        prompt, resolved_prompt_key = get_prompt_text(item, prompt_key)
        return {
            "reference_pixel_values": reference_pixel_values,
            "target_pixel_values": target_pixel_values,
            "pixel_values": target_pixel_values,
            "reference_image_paths": image_paths[:-1],
            "target_image_path": image_paths[-1],
            "prompt": prompt,
            "prompt_type": resolved_prompt_key
        }


# --- 2. Aspect Ratio Sampler ---
class AspectBatchSampler(Sampler):
    def __init__(self, buckets, target_resolutions, batch_size, prompt_keys,
                 num_replicas=1, rank=0, seed=42, shuffle=True):
        self.buckets = buckets
        self.target_resolutions = target_resolutions
        self.batch_size = batch_size
        self.prompt_keys = prompt_keys
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches = []
        for bucket_key, indices in self.buckets.items():
            if self.shuffle:
                shuffled_indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]
            else:
                shuffled_indices = indices

            bucket_idx = bucket_key[0] if isinstance(bucket_key, tuple) else bucket_key
            target_res = self.target_resolutions[bucket_idx]
            for i in range(0, len(shuffled_indices), self.batch_size):
                batch_indices = shuffled_indices[i:i + self.batch_size]
                if len(batch_indices) == self.batch_size:
                    # Select which prompt type to use for this batch at the sampler level
                    selected_prompt_key = random.choice(self.prompt_keys)
                    # Key point: output a list of tuples, where each element is (idx, res, prompt_key)
                    batch_info = [(idx, target_res, selected_prompt_key) for idx in batch_indices]
                    all_batches.append(batch_info)

        if self.shuffle:
            batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_perm]

        total_batches = (len(all_batches) // self.num_replicas) * self.num_replicas
        local_batches = all_batches[self.rank:total_batches:self.num_replicas]

        return iter(local_batches)

    def __len__(self):
        total_valid_batches = sum(len(indices) // self.batch_size for indices in self.buckets.values())
        return total_valid_batches // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch


# --- 3. DataLoader ---
def collate_fn(examples):
    reference_counts = {example["reference_pixel_values"].shape[0] for example in examples}
    if len(reference_counts) != 1:
        raise ValueError("all samples in an edit batch must have the same number of reference images")

    reference_pixel_values = torch.stack([example["reference_pixel_values"] for example in examples])
    target_pixel_values = torch.stack([example["target_pixel_values"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    prompt_types = [example["prompt_type"] for example in examples]
    return {
        "reference_pixel_values": reference_pixel_values,
        "target_pixel_values": target_pixel_values,
        "pixel_values": target_pixel_values,
        "reference_image_paths": [example["reference_image_paths"] for example in examples],
        "target_image_paths": [example["target_image_path"] for example in examples],
        "prompts": prompts,
        "prompt_types": prompt_types
    }


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_sampler, batch_size, **kwargs):
        # Remove potentially conflicting arguments
        kwargs.pop('batch_size', None)
        kwargs.pop('shuffle', None)

        super().__init__(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, **kwargs)
        self._real_batch_size = batch_size

    @property
    def batch_size(self):
        return self._real_batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._real_batch_size = value

