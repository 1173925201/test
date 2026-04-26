import os
import uuid
import requests
from io import BytesIO
from PIL import Image
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import PreparePopupBackgroundInput, PreparePopupBackgroundOutput
from utils.file.file import File
from coze_coding_dev_sdk.s3 import S3SyncStorage


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
        raise Exception(f"背景图片不存在: {url}")

    return Image.open(url)


def prepare_popup_background_node(
    state: PreparePopupBackgroundInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> PreparePopupBackgroundOutput:
    """
    title: 准备弹窗背景
    desc: 标准化调用方传入的弹窗截图背景，并传递给合成节点
    integrations: PIL图像处理, Storage存储
    """
    try:
        background = load_image_file(state.background_image).convert("RGBA")

        temp_path = f"/tmp/popup_background_{uuid.uuid4().hex}.png"
        background.save(temp_path, "PNG", optimize=True)

        storage = S3SyncStorage()
        with open(temp_path, "rb") as f:
            file_key = storage.upload_file(
                file_content=f.read(),
                file_name=f"popup_background_{uuid.uuid4().hex}.png",
                content_type="image/png",
            )

        file_url = storage.generate_presigned_url(key=file_key, expire_time=86400)
        background_image = File(url=file_url, file_type="image")

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return PreparePopupBackgroundOutput(
            cutout_image=state.cutout_image,
            logo_image=state.logo_image,
            background_image=background_image,
        )

    except Exception as e:
        raise Exception(f"弹窗背景准备失败: {str(e)}")
