import os
import random
import uuid
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import GenerateRandomBackgroundInput, GenerateRandomBackgroundOutput
from utils.file.file import File
from coze_coding_dev_sdk.s3 import S3SyncStorage


BACKGROUND_PRESETS = [
    {
        "top": (247, 251, 255),
        "bottom": (205, 222, 239),
        "floor": (226, 232, 238, 255),
        "panel": (255, 255, 255, 92),
        "accent": [(77, 139, 202, 120), (249, 189, 89, 115), (64, 172, 144, 105)],
    },
    {
        "top": (255, 244, 235),
        "bottom": (219, 232, 224),
        "floor": (232, 225, 213, 255),
        "panel": (255, 255, 255, 84),
        "accent": [(207, 93, 78, 115), (47, 130, 118, 110), (244, 183, 84, 120)],
    },
    {
        "top": (237, 241, 247),
        "bottom": (212, 219, 229),
        "floor": (220, 224, 230, 255),
        "panel": (255, 255, 255, 75),
        "accent": [(91, 112, 184, 110), (80, 168, 190, 110), (238, 137, 99, 110)],
    },
    {
        "top": (245, 248, 240),
        "bottom": (211, 229, 216),
        "floor": (226, 234, 220, 255),
        "panel": (255, 255, 255, 88),
        "accent": [(97, 150, 105, 115), (232, 168, 92, 105), (92, 144, 184, 105)],
    },
]


def _interpolate_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3)) + (255,)


def make_vertical_gradient(width: int, height: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    """生成纵向渐变背景"""
    gradient = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(gradient)
    for y in range(height):
        t = y / max(1, height - 1)
        draw.line((0, y, width, y), fill=_interpolate_color(top, bottom, t))
    return gradient


def add_soft_shape(
    base: Image.Image,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int, int],
    blur_radius: int,
) -> None:
    """添加柔和色块，丰富随机背景层次"""
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.ellipse(bbox, fill=color)
    layer = layer.filter(ImageFilter.GaussianBlur(blur_radius))
    base.alpha_composite(layer)


def add_subtle_noise(base: Image.Image, strength: int = 8) -> Image.Image:
    """添加轻微噪声，避免纯色背景过于平面"""
    arr = np.array(base).astype(np.int16)
    noise = np.random.default_rng().integers(-strength, strength + 1, arr[:, :, :3].shape)
    arr[:, :, :3] = np.clip(arr[:, :, :3] + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def make_random_background(width: int, height: int) -> Image.Image:
    """
    生成随机展示背景。
    用程序化背景替代固定截图，保证每次工作流运行都有不同的氛围。
    """
    rng = random.SystemRandom()
    preset = rng.choice(BACKGROUND_PRESETS)
    background = make_vertical_gradient(width, height, preset["top"], preset["bottom"])
    draw = ImageDraw.Draw(background, "RGBA")

    horizon = round(height * rng.uniform(0.47, 0.55))
    draw.rectangle((0, horizon, width, height), fill=preset["floor"])

    panel_count = rng.randint(3, 5)
    panel_gap = round(width * 0.025)
    panel_width = round((width - panel_gap * (panel_count + 1)) / panel_count)
    panel_top = round(height * 0.08)
    panel_bottom = round(horizon * rng.uniform(0.80, 0.92))
    for idx in range(panel_count):
        x0 = panel_gap + idx * (panel_width + panel_gap)
        x1 = x0 + panel_width
        draw.rounded_rectangle(
            (x0, panel_top, x1, panel_bottom),
            radius=28,
            fill=preset["panel"],
            outline=(255, 255, 255, 75),
            width=2,
        )

    for _ in range(rng.randint(5, 9)):
        accent = rng.choice(preset["accent"])
        box_w = rng.randint(round(width * 0.10), round(width * 0.24))
        box_h = rng.randint(round(height * 0.035), round(height * 0.085))
        x0 = rng.randint(round(width * 0.06), max(round(width * 0.07), width - box_w - round(width * 0.06)))
        y0 = rng.randint(round(height * 0.10), max(round(height * 0.11), horizon - box_h - round(height * 0.05)))
        draw.rounded_rectangle(
            (x0, y0, x0 + box_w, y0 + box_h),
            radius=20,
            fill=accent,
        )

    for _ in range(rng.randint(3, 5)):
        color = rng.choice(preset["accent"])
        radius = rng.randint(round(width * 0.12), round(width * 0.28))
        cx = rng.randint(-round(width * 0.10), round(width * 1.10))
        cy = rng.randint(round(height * 0.04), round(height * 0.42))
        add_soft_shape(
            background,
            (cx - radius, cy - radius, cx + radius, cy + radius),
            (color[0], color[1], color[2], 38),
            blur_radius=36,
        )

    floor_line = (255, 255, 255, 68)
    center_x = width // 2
    for offset in range(-4, 5):
        x = center_x + offset * round(width * 0.11)
        draw.line((x, horizon, center_x + offset * round(width * 0.36), height), fill=floor_line, width=2)
    for idx in range(1, 8):
        y = horizon + round((height - horizon) * (idx / 8) ** 1.45)
        draw.line((0, y, width, y), fill=(255, 255, 255, 58), width=2)

    light_layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
    light_draw = ImageDraw.Draw(light_layer, "RGBA")
    for _ in range(rng.randint(2, 4)):
        x = rng.randint(round(width * 0.12), round(width * 0.88))
        light_draw.ellipse(
            (x - 120, -80, x + 120, 120),
            fill=(255, 255, 255, rng.randint(70, 115)),
        )
    light_layer = light_layer.filter(ImageFilter.GaussianBlur(26))
    background.alpha_composite(light_layer)

    return add_subtle_noise(background, strength=5)


def get_output_size() -> tuple[int, int]:
    """使用旧背景图尺寸作为输出画布尺寸，文件缺失时回退到默认尺寸"""
    bg_path = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), "assets/没有logo和人物的网页截图.png")
    if os.path.exists(bg_path):
        with Image.open(bg_path) as img:
            return img.size
    return (1500, 3248)


def generate_random_background_node(
    state: GenerateRandomBackgroundInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> GenerateRandomBackgroundOutput:
    """
    title: 生成随机背景
    desc: 在抠图主体后生成随机展示背景，并传递给合成节点
    integrations: PIL图像处理, Storage存储
    """
    try:
        output_width, output_height = get_output_size()
        background = make_random_background(output_width, output_height)

        temp_path = f"/tmp/random_background_{uuid.uuid4().hex}.png"
        background.save(temp_path, "PNG", optimize=True)

        storage = S3SyncStorage()
        with open(temp_path, "rb") as f:
            file_key = storage.upload_file(
                file_content=f.read(),
                file_name=f"random_background_{uuid.uuid4().hex}.png",
                content_type="image/png",
            )

        file_url = storage.generate_presigned_url(key=file_key, expire_time=86400)
        background_image = File(url=file_url, file_type="image")

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return GenerateRandomBackgroundOutput(
            cutout_image=state.cutout_image,
            logo_image=state.logo_image,
            background_image=background_image,
        )

    except Exception as e:
        raise Exception(f"随机背景生成失败: {str(e)}")
