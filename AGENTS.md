## 项目概述
- **名称**: IP换装展示工作流
- **功能**: 基于固定的两个IP形象（打卡姿势+站着姿势），并行生成IP换装图。站着的直接输出，打卡的经过抠图合成后输出

### 节点清单
| 节点名 | 文件位置 | 类型 | 功能描述 | 分支逻辑 | 配置文件 |
|-------|---------|------|---------|---------|---------|
| start | `graph.py` | task | 虚拟起始节点，触发并行执行 | - | - |
| generate_standing | `nodes/generate_standing_node.py` | task | 生成站着的IP换装图，直接输出 | 并行分支1 | - |
| generate_daka | `nodes/generate_daka_node.py` | task | 生成打卡IP换装图，后续需处理 | 并行分支2 | - |
| resize_image | `nodes/resize_image_node.py` | task | 压缩打卡图片到512px高度 | - | - |
| remove_background | `nodes/remove_background_node.py` | task | 智能抠图，去除打卡图片背景 | - | - |
| prepare_popup_background | `nodes/prepare_popup_background_node.py` | task | 标准化弹窗截图背景 | - | - |
| composite_image | `nodes/composite_image_node.py` | task | 将抠图主体合成到弹窗背景，并添加Logo徽章 | - | - |

**类型说明**: task(task节点) / agent(大模型) / condition(条件分支) / looparray(列表循环) / loopcond(条件循环)

## 工作流流程
```
输入(人物图+Logo+弹窗背景)
    ↓
   start (虚拟起始节点)
    ↓
    ├─────────────────────────────┐
    ↓                             ↓
┌─────────────────┐     ┌─────────────────┐
│ generate_standing│     │  generate_daka  │
│  (站着的IP形象)   │     │  (打卡IP形象)   │
└────────┬────────┘     └────────┬────────┘
         │                       ↓
         │              ┌─────────────────┐
         │              │  resize_image   │
         │              └────────┬────────┘
         │                       ↓
         │              ┌─────────────────┐
         │              │remove_background│
         │              └────────┬────────┘
         │                       ↓
         │              ┌─────────────────┐
         │              │prepare_popup_   │
         │              │   background    │
         │              └────────┬────────┘
         │                       ↓
         │              ┌─────────────────┐
         │              │ composite_image │
         │              └────────┬────────┘
         ↓                       ↓
        END                     END
         │                       │
    standing_image          final_image
    (站着图输出)           (打卡合成图输出)
```

**并行执行说明**：
- `start` 节点同时触发 `generate_standing` 和 `generate_daka`，**真正实现并行执行**
- **分支1**（站着）：生成后直接输出到 `standing_image`，然后结束
- **分支2**（打卡）：生成后继续经过压缩、抠图、弹窗背景准备、合成链路，输出到 `final_image`
- **速度优势**：两个生图任务同时运行，总耗时 ≈ max(生图时间, 打卡链路时间) 而非两者之和

**输入说明**：
- `person_image` (必填): 人物形象照，提供衣服样式
- `logo_image` (必填): Logo图片，会出现在衣服上
- `background_image` (必填): 弹窗截图背景，用于替换顶部IP形象

**输出说明**：
- `standing_image`: 站着的IP换装图片（直接输出，白色背景）
- `final_image`: 打卡姿势的最终合成图片（经过抠图+合成）

**固定的IP基础形象**：
- 打卡姿势: `assets/ip图片_打卡.jpg`
- 站着姿势: `assets/ip图片_站着的.png`

## 弹窗背景合成说明
`prepare_popup_background`节点会在抠图完成后，标准化调用方传入的弹窗截图背景；`composite_image`节点负责把打卡IP换装主体和Logo合成到该背景上。输出画布保持原弹窗背景尺寸。

### 背景处理
1. **来源**：使用工作流入参`background_image`
2. **标准化**：统一转为RGBA PNG并上传存储
3. **尺寸**：保持输入背景原始尺寸

### 人物处理
1. **缩放**：以375x812弹窗为基准，最大宽145px、最大高125px
2. **位置**：覆盖弹窗顶部原小人区域，中心X=187、底部Y=258
3. **适配**：坐标和尺寸按背景实际尺寸等比缩放

### Logo处理
1. **徽章样式**：圆形背景和灰色边框（基准34px）
2. **Logo缩放**：占徽章90%宽度、80%高度
3. **位置**：标题左侧，基准X=112、Y=256

### 合成顺序
1. 弹窗背景图
2. 人物
4. Logo徽章

## 智能抠图逻辑
`remove_background`节点采用专业连通组件分析算法：
1. **边缘采样**：采样边缘12像素确定背景参考色（使用中位数更鲁棒）
2. **多维度判断**：计算颜色距离、饱和度、明度
3. **连通组件分析**：找到与边界相连的背景区域
4. **边缘恢复**：智能恢复可能被误判的主体边缘
5. **最大组件保留**：去除噪点，保留完整主体

**算法参数**：
- 颜色距离阈值：18（针对AI生成的灰色背景优化）
- 亮度阈值：180（灰色背景通常>180）
- 饱和度阈值：45（灰色背景低饱和度）
- 边缘采样宽度：12像素
- 膨胀迭代：2次（恢复边缘）

**优势**：
- 精确分离背景与主体边界
- 保留主体完整细节
- 自适应不同灰度的背景色

## 技能使用
- 节点`generate_standing`和`generate_daka`使用图片生成技能 (doubao-seedream-5-0-260128模型)
  - **多图参考**：同时上传IP形象、人物衣服样式、Logo
  - 两个节点**并行执行**，提升速度
- 节点`remove_background`、`prepare_popup_background`和`composite_image`使用PIL/OpenCV进行图像处理
- 文件存储使用S3SyncStorage

## 资源文件
- `assets/ip图片_打卡.jpg` - 打卡姿势的IP基础形象
- `assets/ip图片_站着的.png` - 站着姿势的IP基础形象
- `assets/logo.png` - Logo图
- `assets/人物.png` - 示例人物图
