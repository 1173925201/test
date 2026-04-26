import os
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from graphs.state import GenerateStandingInput, GenerateStandingOutput, IP_STANDING_PATH
from coze_coding_dev_sdk import ImageGenerationClient
from coze_coding_dev_sdk.s3 import S3SyncStorage
from utils.file.file import File


def _get_image_url(storage: S3SyncStorage, image_path: str) -> str:
    """获取图片的可访问URL，本地路径会自动上传"""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path
    
    if not os.path.isabs(image_path):
        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "")
        image_path = os.path.join(workspace_path, image_path)
    
    if not os.path.exists(image_path):
        raise Exception(f"图片文件不存在: {image_path}")
    
    with open(image_path, "rb") as f:
        file_content = f.read()
    
    file_ext = os.path.splitext(image_path)[1].lower() or ".png"
    import hashlib
    file_hash = hashlib.md5(file_content).hexdigest()[:8]
    file_name = f"ip_standing_{file_hash}{file_ext}"
    
    content_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"
    }
    content_type = content_type_map.get(file_ext, "image/png")
    
    key = storage.upload_file(file_content=file_content, file_name=file_name, content_type=content_type)
    return storage.generate_presigned_url(key=key, expire_time=2592000)


def _upload_user_file(storage: S3SyncStorage, image_file: File) -> str:
    """处理用户上传的文件"""
    url = image_file.url
    if url.startswith("http://") or url.startswith("https://"):
        return url
    
    if not os.path.isabs(url):
        workspace_path = os.getenv("COZE_WORKSPACE_PATH", "")
        url = os.path.join(workspace_path, url)
    
    if not os.path.exists(url):
        raise Exception(f"图片文件不存在: {url}")
    
    with open(url, "rb") as f:
        file_content = f.read()
    
    file_ext = os.path.splitext(url)[1].lower() or ".jpg"
    import hashlib
    file_hash = hashlib.md5(file_content).hexdigest()[:8]
    file_name = f"user_{file_hash}{file_ext}"
    
    content_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"
    }
    content_type = content_type_map.get(file_ext, "image/jpeg")
    
    key = storage.upload_file(file_content=file_content, file_name=file_name, content_type=content_type)
    return storage.generate_presigned_url(key=key, expire_time=2592000)


def generate_standing_node(state: GenerateStandingInput, config: RunnableConfig, runtime: Runtime[Context]) -> GenerateStandingOutput:
    """
    title: 生成站着IP换装图
    desc: 基于站着姿势的IP形象，生成换装图（直接输出，无需后续处理）
    integrations: 图片生成, 对象存储
    """
    ctx = runtime.context
    client = ImageGenerationClient(ctx=ctx)
    
    storage = S3SyncStorage(
        endpoint_url=os.getenv("COZE_BUCKET_ENDPOINT_URL"),
        access_key="",
        secret_key="",
        bucket_name=os.getenv("COZE_BUCKET_NAME"),
        region="cn-beijing",
    )
    
    # 处理图片URL
    person_url = _upload_user_file(storage, state.person_image)
    logo_url = _upload_user_file(storage, state.logo_image)
    standing_ip_url = _get_image_url(storage, IP_STANDING_PATH)
    
    # 生成站着的图
    prompt = (
        "严格依照IP基础形象的样子，包括：外貌特征、发型、面部表情、"
        "身体姿势、手部动作、以及手上拿着的所有物品，"
        "保持IP原图的动作姿态完全不变，"
        "给这个IP形象穿上人物形象中的衣服（参考衣服的款式、颜色和风格），"
        "并在衣服上添加Logo的图案或标识，"
        "生成完整的人物全身像，从头部到脚部都要包含，"
        "人物位于画面中央，主体完整不超出边界，"
        "【背景必须是纯白色RGB(255,255,255)，无渐变无阴影】，"
        "3D卡通渲染风格，高清细节，光线明亮，专业品质"
    )
    
    response = client.generate(
        prompt=prompt,
        image=[standing_ip_url, person_url, logo_url],
        size="1K",
        watermark=False,
        model="doubao-seedream-5-0-260128"
    )
    
    if not response.success:
        raise Exception(f"站着IP换装图生成失败: {response.error_messages}")
    
    standing_image = File(url=response.image_urls[0], file_type="image")
    
    return GenerateStandingOutput(standing_image=standing_image)
