import os
from torch.utils.data import Dataset
import json
import random
from pathlib import Path
from local_paths import resolve_existing_path
from dataset import get_edit_image_paths, get_prompt_text


class TextPromptDataset(Dataset):
    def __init__(self, dataset_path="a.jsonl", prompt_keys = ['short_en', 'detailed_en', 'short_zh', 'detailed_zh'], num_prompts=16,have_gt=False, data_root: str | Path | None = None):
        # only read the first num_prompts lines
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else Path(__file__).resolve().parent
        self.dataset_path = resolve_existing_path(dataset_path, self.data_root)
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line.strip()) for line in f.readlines()]
        self.prompts = []
        self.reference_image_paths = []
        self.target_image_paths = []
        self.have_gt_image = have_gt
        for data in all_data:
            selected_keys =  random.choice(prompt_keys)
            prompt, _ = get_prompt_text(data, selected_keys)
            self.prompts.append(prompt)
            if have_gt:
                image_paths = get_edit_image_paths(data, jsonl_path=self.dataset_path, data_root=self.data_root)
                self.reference_image_paths.append(image_paths[:-1])
                self.target_image_paths.append(image_paths[-1])
            if len(self.prompts) >= num_prompts:
                break

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        if self.have_gt_image:
            return {
                "prompt": self.prompts[idx],
                "reference_image_paths": self.reference_image_paths[idx],
                "target_image_path": self.target_image_paths[idx],
            }
        else:
            return {"prompt": self.prompts[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        if "target_image_path" not in examples[0]:
            return prompts
        reference_image_paths = [example["reference_image_paths"] for example in examples]
        target_image_paths = [example["target_image_path"] for example in examples]
        return prompts, reference_image_paths, target_image_paths
