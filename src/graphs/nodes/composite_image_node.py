import os
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


def composite_image_node(state: CompositeImageInput, config: RunnableConfig, runtime: Runtime[Context]) -> CompositeImageOutput:
    """
    title: 合成最终图片
    desc: 将IP换装人物和Logo合成到背景图上，包含渐变蒙蔽、智能Logo徽章和阴影效果
    integrations: PIL图像处理, Storage存储
    """
    ctx = runtime.context

    try:
        # 加载背景图（本地文件）
        bg_path = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), "assets/没有logo和人物的网页截图.png")
        if not os.path.exists(bg_path):
            raise Exception(f"背景图不存在: {bg_path}")

        background = Image.open(bg_path).convert("RGBA")

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
        temp_path = "/tmp/final_composite.png"
        result.save(temp_path, "PNG", optimize=True)

        storage = S3SyncStorage()
        with open(temp_path, "rb") as f:
            file_key = storage.upload_file(
                file_content=f.read(),
                file_name="final_composite.png",
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
