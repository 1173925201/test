import os
import random
import uuid
import requests
from io import BytesIO
from PIL import Image, ImageChops, ImageDraw, ImageFilter
import numpy as np
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import CompositeImageInput, CompositeImageOutput
from utils.file.file import File
from coze_coding_dev_sdk.s3 import S3SyncStorage


def resize_by_height(image: Image.Image, target_height: int) -> Image.Image:
    """按目标高度等比缩放图片"""
    width, height = image.size
    target_width = round(width * target_height / height)
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def sample_logo_edge_color(logo: Image.Image, edge_width: int = 8) -> tuple[int, int, int, int]:
    """
    采样Logo边缘颜色，用作徽章背景色
    提取边缘可见像素的中位数颜色
    """
    rgba = logo.convert("RGBA")
    arr = np.array(rgba)
    h, w, _ = arr.shape
    ew = max(1, min(edge_width, h // 4, w // 4))

    strips = [
        arr[:ew, :, :].reshape(-1, 4),
        arr[h - ew :, :, :].reshape(-1, 4),
        arr[:, :ew, :].reshape(-1, 4),
        arr[:, w - ew :, :].reshape(-1, 4),
    ]
    edge_pixels = np.concatenate(strips, axis=0)
    visible = edge_pixels[edge_pixels[:, 3] > 0]
    if len(visible) == 0:
        return (255, 255, 255, 255)

    color = np.median(visible[:, :4], axis=0)
    return tuple(int(round(v)) for v in color)


def add_bottom_fade(image: Image.Image, fade_start: float = 0.58, fade_end: float = 0.76) -> Image.Image:
    """
    为图片添加底部渐变透明效果
    从fade_start位置开始渐变，到fade_end位置完全透明
    """
    rgba = image.convert("RGBA")
    width, height = rgba.size
    alpha = rgba.getchannel("A")

    # 创建渐变遮罩
    fade_mask = Image.new("L", (1, height), 255)
    start_y = max(0, min(height - 1, round(height * fade_start)))
    end_y = max(start_y + 1, min(height, round(height * fade_end)))

    # 渐变区域
    for y in range(start_y, end_y):
        progress = (y - start_y) / max(1, end_y - start_y)
        fade_mask.putpixel((0, y), round(255 * (1.0 - progress)))

    # 渐变结束后完全透明
    for y in range(end_y, height):
        fade_mask.putpixel((0, y), 0)

    # 拉伸到全宽并应用
    fade_mask = fade_mask.resize((width, height), Image.Resampling.BILINEAR)
    merged_alpha = Image.new("L", (width, height))
    merged_alpha.paste(ImageChops.multiply(alpha, fade_mask))
    rgba.putalpha(merged_alpha)
    return rgba


def make_logo_badge(
    logo: Image.Image,
    badge_size: int = 82,
    logo_width_ratio: float = 0.9,
    logo_height_ratio: float = 0.58,
) -> Image.Image:
    """
    制作圆形Logo徽章
    - 使用Logo边缘颜色作为徽章背景
    - Logo居中裁剪到圆形
    - 灰色边框
    """
    badge = Image.new("RGBA", (badge_size, badge_size), (0, 0, 0, 0))
    
    # 采样Logo边缘颜色作为填充色
    fill_color = sample_logo_edge_color(logo)

    # 绘制圆形填充
    circle_fill = Image.new("RGBA", (badge_size, badge_size), (0, 0, 0, 0))
    draw_fill = ImageDraw.Draw(circle_fill)
    draw_fill.ellipse((1, 1, badge_size - 2, badge_size - 2), fill=fill_color)

    # 绘制圆形边框
    circle_outline = Image.new("RGBA", (badge_size, badge_size), (0, 0, 0, 0))
    draw_outline = ImageDraw.Draw(circle_outline)
    draw_outline.ellipse((1, 1, badge_size - 2, badge_size - 2), outline=(170, 170, 170, 255), width=2)

    # 创建圆形裁剪蒙版（用于裁剪Logo）
    clip_mask = Image.new("L", (badge_size, badge_size), 0)
    draw_mask = ImageDraw.Draw(clip_mask)
    draw_mask.ellipse((3, 3, badge_size - 4, badge_size - 4), fill=255)

    # 准备Logo（居中缩放）
    inner = Image.new("RGBA", (badge_size, badge_size), (0, 0, 0, 0))
    inner_logo = logo.convert("RGBA")
    inner_logo.thumbnail(
        (round(badge_size * logo_width_ratio), round(badge_size * logo_height_ratio)),
        Image.Resampling.LANCZOS,
    )
    offset = ((badge_size - inner_logo.width) // 2, (badge_size - inner_logo.height) // 2)
    inner.paste(inner_logo, offset, inner_logo)
    
    # 裁剪Logo到圆形
    inner_alpha = inner.getchannel("A")
    inner.putalpha(ImageChops.multiply(inner_alpha, clip_mask))

    # 合成徽章（填充 -> Logo -> 边框）
    badge.alpha_composite(circle_fill)
    badge.alpha_composite(inner)
    badge.alpha_composite(circle_outline)
    return badge


def add_shadow(image: Image.Image, blur_radius: int = 10, opacity: int = 70) -> Image.Image:
    """为图片添加投影效果"""
    alpha = image.getchannel("A")
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda v: min(255, round(v * opacity / 255))))
    return shadow.filter(ImageFilter.GaussianBlur(blur_radius))


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

    # 后墙大面板
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

    # 随机海报/色块，像展厅或活动背景板，但不生成文字
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

    # 柔和光斑
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

    # 地面透视线，增强主体落地感
    floor_line = (255, 255, 255, 68)
    center_x = width // 2
    for offset in range(-4, 5):
        x = center_x + offset * round(width * 0.11)
        draw.line((x, horizon, center_x + offset * round(width * 0.36), height), fill=floor_line, width=2)
    for idx in range(1, 8):
        y = horizon + round((height - horizon) * (idx / 8) ** 1.45)
        draw.line((0, y, width, y), fill=(255, 255, 255, 58), width=2)

    # 顶部柔光
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


def composite_image_node(state: CompositeImageInput, config: RunnableConfig, runtime: Runtime[Context]) -> CompositeImageOutput:
    """
    title: 合成最终图片
    desc: 将IP换装人物和Logo合成到背景图上，包含渐变蒙蔽、智能Logo徽章和阴影效果
    integrations: PIL图像处理, Storage存储
    """
    ctx = runtime.context

    try:
        # 生成随机背景
        output_width, output_height = get_output_size()
        background = make_random_background(output_width, output_height)

        # 加载人物图（从URL）
        person_response = requests.get(state.cutout_image.url, timeout=30)
        person_response.raise_for_status()
        person = Image.open(BytesIO(person_response.content)).convert("RGBA")

        # 加载Logo（使用传入的logo_image，动态调整）
        try:
            logo_response = requests.get(state.logo_image.url, timeout=30)
            logo_response.raise_for_status()
            logo = Image.open(BytesIO(logo_response.content)).convert("RGBA")
        except Exception:
            # 如果URL下载失败，使用本地logo作为fallback
            logo_path = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), "assets/logo.png")
            if os.path.exists(logo_path):
                logo = Image.open(logo_path).convert("RGBA")
            else:
                raise Exception(f"无法加载Logo图片")

        # 1. 处理人物
        # 按高度缩放（目标高度700像素）
        person = resize_by_height(person, 700)
        # 添加底部渐变（从57%到67%位置渐变）
        person = add_bottom_fade(person, fade_start=0.57, fade_end=0.67)
        # 创建阴影
        person_shadow = add_shadow(person, blur_radius=12, opacity=68)

        # 2. 处理Logo
        # 制作圆形徽章（150像素），Logo占满90%宽高
        logo_badge = make_logo_badge(
            logo,
            badge_size=150,
            logo_width_ratio=0.9,  # 占满宽度
            logo_height_ratio=0.8,  # 占满高度
        )

        # 3. 合成到背景
        result = background.copy()

        # 人物位置（居中，y=680）
        person_x = (result.width - person.width) // 2
        person_y = 680

        # 阴影位置（稍微偏移）
        shadow_x = person_x
        shadow_y = person_y + 14

        # 先粘贴阴影，再粘贴人物
        result.alpha_composite(person_shadow, (shadow_x, shadow_y))
        result.alpha_composite(person, (person_x, person_y))

        # Logo位置（x=320, y=1150）
        logo_x = 320
        logo_y = 1150
        result.alpha_composite(logo_badge, (logo_x, logo_y))

        # 4. 保存并上传
        temp_path = f"/tmp/final_composite_{uuid.uuid4().hex}.png"
        result.save(temp_path, "PNG", optimize=True)

        storage = S3SyncStorage()
        with open(temp_path, "rb") as f:
            file_key = storage.upload_file(
                file_content=f.read(),
                file_name=f"final_composite_{uuid.uuid4().hex}.png",
                content_type="image/png"
            )

        file_url = storage.generate_presigned_url(key=file_key, expire_time=86400)
        final_image = File(url=file_url, file_type="image")

        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return CompositeImageOutput(final_image=final_image)

    except Exception as e:
        raise Exception(f"图片合成失败: {str(e)}")
