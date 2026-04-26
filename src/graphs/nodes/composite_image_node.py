import os
import uuid
import requests
from io import BytesIO
from PIL import Image, ImageChops, ImageDraw
import numpy as np
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import CompositeImageInput, CompositeImageOutput
from utils.file.file import File
from coze_coding_dev_sdk.s3 import S3SyncStorage


BASE_POPUP_WIDTH = 375
BASE_POPUP_HEIGHT = 812
IP_CENTER_X = 187
IP_BOTTOM_Y = 258
IP_MAX_WIDTH = 145
IP_MAX_HEIGHT = 125
LOGO_BADGE_X = 112
LOGO_BADGE_Y = 256
LOGO_BADGE_SIZE = 34


def load_image_file(image_file: File, timeout: int = 30) -> Image.Image:
    """加载URL或本地路径图片"""
    url = image_file.url
    if url.startswith("http://") or url.startswith("https://"):
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))

    if not os.path.isabs(url):
        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "")
        url = os.path.join(workspace_path, url)

    if not os.path.exists(url):
        raise Exception(f"图片不存在: {url}")

    return Image.open(url)


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


def scale_popup_value(value: int, scale: float) -> int:
    """按弹窗截图尺寸缩放坐标或尺寸"""
    return max(1, round(value * scale))


def fit_ip_to_popup(image: Image.Image, background_size: tuple[int, int]) -> Image.Image:
    """把IP缩放到弹窗顶部mascot区域"""
    width, height = background_size
    max_width = scale_popup_value(IP_MAX_WIDTH, width / BASE_POPUP_WIDTH)
    max_height = scale_popup_value(IP_MAX_HEIGHT, height / BASE_POPUP_HEIGHT)
    ip = image.convert("RGBA").copy()
    ip.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return ip


def composite_image_node(state: CompositeImageInput, config: RunnableConfig, runtime: Runtime[Context]) -> CompositeImageOutput:
    """
    title: 合成最终图片
    desc: 将IP换装人物和Logo合成到背景图上，包含渐变蒙蔽、智能Logo徽章和阴影效果
    integrations: PIL图像处理, Storage存储
    """
    ctx = runtime.context

    try:
        # 加载弹窗截图背景（由上游节点标准化）
        background = load_image_file(state.background_image).convert("RGBA")

        # 加载人物图（从URL）
        person = load_image_file(state.cutout_image).convert("RGBA")

        # 加载Logo（使用传入的logo_image，动态调整）
        logo = load_image_file(state.logo_image).convert("RGBA")

        # 1. 处理人物
        person = fit_ip_to_popup(person, background.size)

        # 2. 处理Logo
        scale_x = background.width / BASE_POPUP_WIDTH
        scale_y = background.height / BASE_POPUP_HEIGHT
        logo_badge_size = scale_popup_value(LOGO_BADGE_SIZE, min(scale_x, scale_y))
        logo_badge = make_logo_badge(
            logo,
            badge_size=logo_badge_size,
            logo_width_ratio=0.9,
            logo_height_ratio=0.8,
        )

        # 3. 合成到背景
        result = background.copy()

        # IP位置：弹窗顶部居中，覆盖原小人区域
        person_center_x = round(IP_CENTER_X * scale_x)
        person_bottom_y = round(IP_BOTTOM_Y * scale_y)
        person_x = round(person_center_x - person.width / 2)
        person_y = round(person_bottom_y - person.height)
        result.alpha_composite(person, (person_x, person_y))

        # Logo徽章位置：标题左侧
        logo_x = round(LOGO_BADGE_X * scale_x)
        logo_y = round(LOGO_BADGE_Y * scale_y)
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
