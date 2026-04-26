from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from langgraph.graph import StateGraph, END
from graphs.state import (
    GlobalState,
    GraphInput,
    GraphOutput,
    StartNodeInput,
    StartNodeOutput
)
from graphs.nodes.generate_standing_node import generate_standing_node
from graphs.nodes.generate_daka_node import generate_daka_node
from graphs.nodes.resize_image_node import resize_image_node
from graphs.nodes.remove_background_node import remove_background_node
from graphs.nodes.prepare_popup_background_node import prepare_popup_background_node
from graphs.nodes.composite_image_node import composite_image_node


# 虚拟起始节点：用于触发并行执行
def start_node(state: StartNodeInput, config: RunnableConfig, runtime: Runtime[Context]) -> StartNodeOutput:
    """
    title: 开始节点
    desc: 虚拟起始节点，用于同时触发两个生图节点并行执行
    """
    return StartNodeOutput(
        person_image=state.person_image,
        logo_image=state.logo_image,
        background_image=state.background_image
    )


# 创建状态图, 指定工作流的入参、出参和全局状态
builder = StateGraph(GlobalState, input_schema=GraphInput, output_schema=GraphOutput)

# 添加所有节点
builder.add_node("start", start_node)
builder.add_node("generate_standing", generate_standing_node)
builder.add_node("generate_daka", generate_daka_node)
builder.add_node("resize_image", resize_image_node)
builder.add_node("remove_background", remove_background_node)
builder.add_node("prepare_popup_background", prepare_popup_background_node)
builder.add_node("composite_image", composite_image_node)

# 设置入口点
builder.set_entry_point("start")

# ===== 并行执行：从start同时触发两个生图节点 =====
# start -> generate_standing (并行分支1)
# start -> generate_daka (并行分支2)
builder.add_edge("start", "generate_standing")
builder.add_edge("start", "generate_daka")

# ===== 分支1: 站着的图 -> 直接结束（输出到standing_image）=====
builder.add_edge("generate_standing", END)


# ===== 分支2: 打卡的图 -> 继续处理链路 =====
builder.add_edge("generate_daka", "resize_image")
builder.add_edge("resize_image", "remove_background")
builder.add_edge("remove_background", "prepare_popup_background")
builder.add_edge("prepare_popup_background", "composite_image")
builder.add_edge("composite_image", END)

# 编译图
main_graph = builder.compile()
