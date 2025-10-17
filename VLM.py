from typing import List, Union, Tuple
from PIL import Image
import numpy as np
import cv2
import torch
import re                                   # <-- NEW: dùng để tách thinking/answer
from transformers import AutoProcessor, Glm4vForConditionalGeneration  # đúng tên lớp

class GLM41VClient:
    def __init__(
        self,
        model_id: str = "zai-org/GLM-4.1V-9B-Thinking",
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto"
    ):
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,                 # dùng dtype thay vì torch_dtype
            device_map=device_map,
            trust_remote_code=True
        ).eval()

    @staticmethod
    def _to_pil(img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, str):
            return Image.open(img).convert("RGB")
        if isinstance(img, np.ndarray):
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        raise TypeError("Unsupported image type for VLM input.")

    def generate_text(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        instruction: str,
        system_prompt: str = "You are a vision-language expert. Be concise and precise.",
        max_new_tokens: int = 400,
        temperature: float = 0.2
    ) -> str:
        pil_imgs = [self._to_pil(im) for im in images]

        # Tạo chat messages: nhiều ảnh + 1 text instruction
        content = [{"type": "image", "image": im} for im in pil_imgs]
        content.append({"type": "text", "text": instruction})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",   "content": content},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                do_sample=(temperature > 0),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        # Cắt phần sinh thêm để lấy câu trả lời
        prompt_len = inputs["input_ids"].shape[1]
        gen_only = out_ids[0][prompt_len:]
        text = self.processor.decode(gen_only, skip_special_tokens=True)
        return text.strip()


# -------------------- NEW: tách thinking / answer --------------------
def split_thinking_answer(text: str) -> Tuple[str, str]:
    """
    Tách thinking và answer từ chuỗi đầu ra của VLM.
    Trả về (thinking, answer). Nếu không nhận diện được 'thinking',
    trả ("", text).
    Hỗ trợ các pattern phổ biến:
      - <think>...</think> [<answer>...</answer>] (hoặc <thinking> ... </thinking>)
      - header kiểu 'Reasoning:/Answer:' hoặc 'Analysis:/Final Answer:'
      - ngăn cách bằng đường kẻ --- / ===
    """
    s = text.strip()

    # 1) Thẻ think/thinking + (tuỳ chọn) answer
    m = re.search(r"<think(?:ing)?>\s*(.*?)\s*</think(?:ing)?>\s*(.*)$",
                  s, flags=re.S | re.I)
    if m:
        thinking = m.group(1).strip()
        tail = m.group(2).strip()
        # nếu còn thẻ answer ở phần tail thì bóc tiếp
        m2 = re.search(r"<answer>\s*(.*?)\s*</answer>\s*$", tail, flags=re.S | re.I)
        answer = m2.group(1).strip() if m2 else tail
        return thinking, answer

    # 2) Header Reasoning/Answer
    m = re.search(
        r"(?:Reasoning|Chain[- ]?of[- ]?Thought|Thinking|Analysis)\s*:\s*(.*?)\s*(?:\n|\r)+\s*(?:Final Answer|Answer)\s*:\s*(.*)$",
        s, flags=re.S | re.I)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # 3) Đường kẻ --- hoặc ===
    m = re.search(r"^(.*?)\n[-=]{3,}\n+(.*)$", s, flags=re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # 4) Không có thinking riêng
    return "", s
# --------------------------------------------------------------------


if __name__ == "__main__":
    import argparse, sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description="GLM-4.1V-9B-Thinking CLI (multi-image + instruction)")
    ap.add_argument("--images", required=True,
                    help="Một hoặc nhiều đường dẫn ảnh, phân tách bằng dấu phẩy. Ví dụ: 'p0.jpg,acc1.png,acc2.jpg'")
    ap.add_argument("--instruction", default=(
        "Describe the person and each visible accessory: type, color/material, logos/brand if visible. Be concise."
    ), help="Instruction gửi cho VLM")
    ap.add_argument("--system", default="You are a vision-language expert. Be concise and precise.",
                    help="System prompt")
    ap.add_argument("--model-id", default="zai-org/GLM-4.1V-9B-Thinking",
                    help="Hugging Face model id")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="Giới hạn token sinh thêm")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Nhiệt độ giải mã (0–1)")
    ap.add_argument("--output", default="",
                    help="Đường dẫn cơ sở để lưu. Sẽ tạo 2 file: *.thinking.txt và *.answer.txt")
    args = ap.parse_args()

    img_paths = [p.strip() for p in args.images.split(",") if p.strip()]
    if not img_paths:
        sys.exit("No images provided.")
    for p in img_paths:
        if not Path(p).exists():
            sys.exit(f"Image not found: {p}")

    client = GLM41VClient(model_id=args.model_id)
    full_text = client.generate_text(
        images=img_paths,
        instruction=args.instruction,
        system_prompt=args.system,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    # Tách thinking / answer
    thinking, answer = split_thinking_answer(full_text)

    # In ra console để bạn thấy nhanh
    if thinking:
        print("===== THINKING =====")
        print(thinking)
    print("===== ANSWER =====")
    print(answer)

    # Ghi file nếu cung cấp --output
    if args.output:
        base = Path(args.output)
        base.parent.mkdir(parents=True, exist_ok=True)
        # tạo tên hai file: <base>.thinking.txt và <base>.answer.txt
        base_str = str(base)
        if base.suffix:                   # nếu có .txt hoặc .log...
            base_str = base_str[: -len(base.suffix)]
        think_p = Path(base_str + ".thinking.txt")
        ans_p   = Path(base_str + ".answer.txt")

        think_p.write_text(thinking + "\n", encoding="utf-8")
        ans_p.write_text(answer + "\n", encoding="utf-8")

        print(f"[SAVED] thinking: {think_p}")
        print(f"[SAVED] answer:   {ans_p}")
