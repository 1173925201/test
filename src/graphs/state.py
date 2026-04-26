import os
from typing import Optional
from pydantic import BaseModel, Field
from utils.file.file import File

# 固定的IP基础形象路径
IP_DAKA_PATH = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), "assets/ip图片_打卡.jpg")
IP_STANDING_PATH = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), "assets/ip图片_站着的.png")


# ==================== 全局状态 ====================
class GlobalState(BaseModel):
    """全局状态定义 - 贯穿整个工作流的数据"""
    person_image: File = Field(..., description="人物形象照（提供衣服样式）")
    logo_image: File = Field(..., description="Logo图片")
    background_image: File = Field(..., description="弹窗截图背景图")
    # 站着姿势相关
    standing_image: Optional[File] = Field(default=None, description="生成的站着的IP换装图（直接输出）")
    # 打卡姿势相关
    daka_image: Optional[File] = Field(default=None, description="生成的打卡IP换装图（白色背景）")
    cutout_image: Optional[File] = Field(default=None, description="抠图后的打卡图片主体")
    final_image: Optional[File] = Field(default=None, description="最终合成的打卡图片")


# ==================== 图的输入输出 ====================
class GraphInput(BaseModel):
    """工作流的输入"""
    person_image: File = Field(..., description="人物形象照（提供衣服样式）")
    logo_image: File = Field(..., description="Logo图片")
    background_image: File = Field(..., description="弹窗截图背景图")


class GraphOutput(BaseModel):
    """工作流的输出"""
    standing_image: File = Field(..., description="站着的IP换装图片（直接输出）")
    final_image: File = Field(..., description="打卡姿势的最终合成图片")


# ==================== 节点1: 生成站着IP换装图 ====================
class GenerateStandingInput(BaseModel):
    """生成站着IP换装节点的输入"""
    person_image: File = Field(..., description="人物形象照（提供衣服样式）")
    logo_image: File = Field(..., description="Logo图片（出现在衣服上）")


class GenerateStandingOutput(BaseModel):
    """生成站着IP换装节点的输出"""
    standing_image: File = Field(..., description="站着的IP换装图")


# ==================== 节点2: 生成打卡IP换装图 ====================
class GenerateDakaInput(BaseModel):
    """生成打卡IP换装节点的输入"""
    person_image: File = Field(..., description="人物形象照（提供衣服样式）")
    logo_image: File = Field(..., description="Logo图片（出现在衣服上）")


class GenerateDakaOutput(BaseModel):
    """生成打卡IP换装节点的输出"""
    daka_image: File = Field(..., description="打卡姿势的IP换装图")
    logo_image: File = Field(..., description="Logo图片（传递到后续节点）")


# ==================== 节点3: 压缩图片分辨率 ====================
class ResizeImageInput(BaseModel):
    """压缩图片节点的输入"""
    daka_image: File = Field(..., description="打卡姿势的IP换装图")
    logo_image: File = Field(..., description="Logo图片")


class ResizeImageOutput(BaseModel):
    """压缩图片节点的输出"""
    daka_image: File = Field(..., description="压缩后的打卡图片（高度512px）")
    logo_image: File = Field(..., description="Logo图片")


# ==================== 节点4: 抠图提取主体 ====================
class RemoveBackgroundInput(BaseModel):
    """抠图节点的输入"""
    daka_image: File = Field(..., description="压缩后的打卡IP换装图")
    logo_image: File = Field(..., description="Logo图片")


class RemoveBackgroundOutput(BaseModel):
    """抠图节点的输出"""
    cutout_image: File = Field(..., description="抠图后的透明背景打卡主体图片")
    logo_image: File = Field(..., description="Logo图片")


# ==================== 节点5: 准备弹窗背景 ====================
class PreparePopupBackgroundInput(BaseModel):
    """弹窗背景节点的输入"""
    cutout_image: File = Field(..., description="抠图后的打卡主体图片")
    logo_image: File = Field(..., description="Logo图片")
    background_image: File = Field(..., description="弹窗截图背景图")


class PreparePopupBackgroundOutput(BaseModel):
    """弹窗背景节点的输出"""
    cutout_image: File = Field(..., description="抠图后的打卡主体图片")
    logo_image: File = Field(..., description="Logo图片")
    background_image: File = Field(..., description="标准化后的弹窗截图背景图")


# ==================== 节点6: 合成最终图片 ====================
class CompositeImageInput(BaseModel):
    """合成图片节点的输入"""
    cutout_image: File = Field(..., description="抠图后的打卡主体图片")
    logo_image: File = Field(..., description="Logo图片")
    background_image: File = Field(..., description="弹窗截图背景图")


class CompositeImageOutput(BaseModel):
    """合成图片节点的输出"""
    final_image: File = Field(..., description="最终合成的打卡图片")


# ==================== 虚拟起始节点 ====================
class StartNodeInput(BaseModel):
    """起始节点的输入"""
    person_image: File = Field(..., description="人物形象照")
    logo_image: File = Field(..., description="Logo图片")
    background_image: File = Field(..., description="弹窗截图背景图")


class StartNodeOutput(BaseModel):
    """起始节点的输出"""
    person_image: File = Field(..., description="人物形象照（传递到下游节点）")
    logo_image: File = Field(..., description="Logo图片（传递到下游节点）")
    background_image: File = Field(..., description="弹窗截图背景图（传递到下游节点）")
