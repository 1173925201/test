from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import ResizeImageInput, ResizeImageOutput
from utils.file.file import File
from coze_coding_dev_sdk.s3 import S3SyncStorage
import requests
from io import BytesIO
from PIL import Image
import os


def resize_image_node(state: ResizeImageInput, config: RunnableConfig, runtime: Runtime[Context]) -> ResizeImageOutput:
    """
    title: 压缩图片分辨率
    desc: 将生成的IP换装图压缩到合适尺寸（512px高度），加速后续处理，人眼看起来仍然清晰
    integrations: PIL图像处理, Storage存储
    """
    ctx = runtime.context

    try:
        # 下载原图
        response = requests.get(state.daka_image.url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        
        original_size = img.size
        
        # 转换为RGB（如果是RGBA）
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # 计算目标尺寸：高度512px，宽度等比缩放
        target_height = 512
        width, height = img.size
        
        if height > target_height:
            ratio = target_height / height
            target_width = int(width * ratio)
            img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            img_resized = img
            target_width, target_height = width, height
        
        # 保存压缩后的图片
        temp_path = "/tmp/resized_image.jpg"
        img_resized.save(temp_path, "JPEG", quality=85, optimize=True)
        
        # 上传
        storage = S3SyncStorage()
        with open(temp_path, "rb") as f:
            file_key = storage.upload_file(
                file_content=f.read(),
                file_name="resized_image.jpg",
                content_type="image/jpeg"
            )
        
        file_url = storage.generate_presigned_url(key=file_key, expire_time=86400)
        resized_image = File(url=file_url, file_type="image")
        
        # 清理
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print(f"图片压缩: {original_size} -> ({target_width}, {target_height})")
        
        # 覆盖daka_image，确保下游抠图使用压缩后的图片
        return ResizeImageOutput(
            daka_image=resized_image,
            logo_image=state.logo_image
        )

    except Exception as e:
        raise Exception(f"图片压缩失败: {str(e)}")
