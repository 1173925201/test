import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import RemoveBackgroundInput, RemoveBackgroundOutput
from utils.file.file import File
from coze_coding_dev_sdk.s3 import S3SyncStorage


def sample_edge_pixels(arr: np.ndarray, edge_width: int) -> np.ndarray:
    """采样边缘像素作为背景参考"""
    h, w, _ = arr.shape
    ew = max(1, min(edge_width, h // 4, w // 4))
    strips = [
        arr[:ew, :, :].reshape(-1, 3),
        arr[h - ew :, :, :].reshape(-1, 3),
        arr[:, :ew, :].reshape(-1, 3),
        arr[:, w - ew :, :].reshape(-1, 3),
    ]
    return np.concatenate(strips, axis=0)


def rgb_to_saturation_and_value(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """计算RGB的饱和度和明度"""
    arr_f = arr.astype(np.float32)
    max_c = arr_f.max(axis=2)
    min_c = arr_f.min(axis=2)
    saturation = np.zeros_like(max_c)
    nonzero = max_c > 0
    saturation[nonzero] = ((max_c[nonzero] - min_c[nonzero]) / max_c[nonzero]) * 255.0
    return saturation, max_c


def make_background_mask(
    arr: np.ndarray,
    edge_width: int,
    distance_threshold: float,
    brightness_threshold: int,
    max_saturation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    创建背景掩码
    使用OpenCV的connectedComponents进行连通组件分析
    """
    h, w, _ = arr.shape
    edge_pixels = sample_edge_pixels(arr, edge_width)
    bg_color = np.median(edge_pixels, axis=0)

    diff = arr.astype(np.float32) - bg_color.astype(np.float32)
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    saturation, value = rgb_to_saturation_and_value(arr)

    candidate = (
        (distance <= distance_threshold)
        & (value >= brightness_threshold)
        & (saturation <= max_saturation)
    )

    # 使用OpenCV进行连通组件分析（更快）
    labels_count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=4)
    
    # 找到与边界相连的标签
    border_labels = np.unique(
        np.concatenate([
            labels[0, :],
            labels[h - 1, :],
            labels[:, 0],
            labels[:, w - 1],
        ])
    )
    
    background = np.isin(labels, border_labels) & candidate
    return background, distance, saturation, value


def dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """膨胀操作（OpenCV实现）"""
    kernel = np.ones((3, 3), dtype=np.uint8)
    grown = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)
    return grown.astype(bool)


def recover_bright_subject_edges(
    foreground: np.ndarray,
    distance: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
    distance_threshold: float,
    brightness_threshold: int,
    max_saturation: int,
) -> np.ndarray:
    """恢复明亮的主体边缘"""
    expanded = dilate(foreground, iterations=2)
    recoverable = (
        expanded
        & (~foreground)
        & (
            (distance > distance_threshold * 0.42)
            | (saturation > max_saturation * 1.45)
            | (value < brightness_threshold + 18)
        )
    )
    return foreground | recoverable


def largest_component(mask: np.ndarray) -> np.ndarray:
    """保留最大的连通组件（主体）"""
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=4,
    )
    if labels_count <= 1:
        return np.zeros_like(mask, dtype=bool)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return labels == largest_label


def remove_background_fast(img: Image.Image) -> Image.Image:
    """
    快速背景去除（使用OpenCV）
    基于连通组件分析，精确分离背景与主体
    """
    # 转换为RGB
    if img.mode != 'RGB':
        img_rgb = img.convert('RGB')
    else:
        img_rgb = img
    
    arr = np.array(img_rgb)
    
    # 参数设置
    edge_width = 12
    distance_threshold = 15.0
    brightness_threshold = 185
    max_saturation = 38
    
    # 创建背景掩码
    background, distance, saturation, value = make_background_mask(
        arr=arr,
        edge_width=edge_width,
        distance_threshold=distance_threshold,
        brightness_threshold=brightness_threshold,
        max_saturation=max_saturation,
    )

    # 前景 = 非背景
    foreground = ~background
    
    # 保留最大连通组件
    foreground = largest_component(foreground)
    
    # 恢复主体边缘
    foreground = recover_bright_subject_edges(
        foreground=foreground,
        distance=distance,
        saturation=saturation,
        value=value,
        distance_threshold=distance_threshold,
        brightness_threshold=brightness_threshold,
        max_saturation=max_saturation,
    )
    
    # 再次保留最大组件
    foreground = largest_component(foreground)

    # 创建RGBA图像
    rgba = np.dstack([arr, (foreground * 255).astype(np.uint8)])
    
    return Image.fromarray(rgba, 'RGBA')


def remove_background_node(state: RemoveBackgroundInput, config: RunnableConfig, runtime: Runtime[Context]) -> RemoveBackgroundOutput:
    """
    title: 抠图提取主体
    desc: 使用OpenCV连通组件分析快速去除背景，1-2秒完成处理
    integrations: OpenCV, PIL图像处理, Storage存储
    """
    ctx = runtime.context

    try:
        # 下载图片
        response = requests.get(state.daka_image.url, timeout=30)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))

        # 使用OpenCV快速抠图
        result_img = remove_background_fast(img)

        # 保存并上传
        temp_path = "/tmp/cutout_image.png"
        result_img.save(temp_path, "PNG", optimize=True)

        storage = S3SyncStorage()
        with open(temp_path, "rb") as f:
            file_key = storage.upload_file(
                file_content=f.read(),
                file_name="cutout_image.png",
                content_type="image/png"
            )

        file_url = storage.generate_presigned_url(key=file_key, expire_time=86400)
        cutout_image = File(url=file_url, file_type="image")

        if os.path.exists(temp_path):
            os.remove(temp_path)

        # 返回抠图结果和logo（传递给后续节点）
        return RemoveBackgroundOutput(
            cutout_image=cutout_image,
            logo_image=state.logo_image
        )

    except Exception as e:
        raise Exception(f"抠图处理失败: {str(e)}")
