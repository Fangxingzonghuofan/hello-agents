"""
智能搜索助手 - 基于 LangGraph + Tavily API 的真实搜索系统
1. 理解用户需求
2. 使用Tavily API真实搜索信息  
3. 生成基于搜索结果的回答
"""

import asyncio
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from tavily import TavilyClient

# 加载环境变量 - 从.env文件读取API密钥等配置
load_dotenv()

# 定义状态结构 - 使用TypedDict为状态提供类型提示
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话消息列表，使用add_messages进行特殊合并处理
    user_query: str        # 用户原始查询
    search_query: str      # 优化后的搜索查询
    search_results: str    # Tavily搜索结果
    final_answer: str      # 最终答案
    step: str             # 当前步骤，用于跟踪工作流进度

# 初始化语言模型客户端 - 配置OpenAI或兼容API
llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL_ID", "gpt-4o-mini"),  # 模型ID，默认使用gpt-4o-mini
    api_key=os.getenv("LLM_API_KEY"),  # API密钥，从环境变量读取
    base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),  # API基础URL，支持自定义
    temperature=0.7  # 生成随机性，0.7是平衡值
)

# 初始化Tavily搜索客户端 - 用于执行真实网络搜索
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def understand_query_node(state: SearchState) -> SearchState:
    """步骤1：理解用户查询并生成搜索关键词"""
    
    # 获取最新的用户消息 - 从消息历史中查找最后一条HumanMessage
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    
    # 构建系统提示，让AI分析用户需求并生成搜索词
    understand_prompt = f"""分析用户的查询："{user_message}"

请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]"""

    # 调用语言模型生成分析结果
    response = llm.invoke([SystemMessage(content=understand_prompt)])
    
    # 提取搜索关键词 - 从AI响应中解析结构化信息
    response_text = response.content
    search_query = user_message  # 默认使用原始查询作为搜索词
    
    # 尝试从AI响应中提取"搜索词："或"搜索关键词："后面的内容
    if "搜索词：" in response_text:
        search_query = response_text.split("搜索词：")[1].strip()
    elif "搜索关键词：" in response_text:
        search_query = response_text.split("搜索关键词：")[1].strip()
    
    # 返回状态更新 - 框架会自动合并到全局状态
    return {
        "user_query": response.content,  # AI的完整分析响应
        "search_query": search_query,    # 提取出的搜索关键词
        "step": "understood",           # 更新步骤标识
        "messages": [AIMessage(content=f"我理解您的需求：{response.content}")]  # 添加AI回复消息
    }

def tavily_search_node(state: SearchState) -> SearchState:
    """步骤2：使用Tavily API进行真实搜索"""
    
    search_query = state["search_query"]  # 从上一步获取优化后的搜索词
    
    try:
        print(f"🔍 正在搜索: {search_query}")
        
        # 调用Tavily搜索API - 执行真实网络搜索
        response = tavily_client.search(
            query=search_query,  # 搜索关键词
            search_depth="basic",  # 搜索深度：basic或advanced
            include_answer=True,  # 包含AI生成的综合答案
            include_raw_content=False,  # 不包含原始HTML内容
            max_results=5  # 最大结果数
        )
        
        # 处理搜索结果 - 格式化Tavily返回的数据
        search_results = ""
        
        # 优先使用Tavily的综合答案（如果提供）
        if response.get("answer"):
            search_results = f"综合答案：\n{response['answer']}\n\n"
        
        # 添加具体的搜索结果条目
        if response.get("results"):
            search_results += "相关信息：\n"
            for i, result in enumerate(response["results"][:3], 1):  # 只取前3个结果
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                search_results += f"{i}. {title}\n{content}\n来源：{url}\n\n"
        
        # 如果没有任何搜索结果
        if not search_results:
            search_results = "抱歉，没有找到相关信息。"
        
        # 返回状态更新
        return {
            "search_results": search_results,  # 格式化后的搜索结果
            "step": "searched",  # 更新步骤标识
            "messages": [AIMessage(content=f"✅ 搜索完成！找到了相关信息，正在为您整理答案...")]  # 添加状态消息
        }
        
    except Exception as e:
        # 搜索失败时的错误处理
        error_msg = f"搜索时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        
        return {
            "search_results": f"搜索失败：{error_msg}",  # 记录错误信息
            "step": "search_failed",  # 特殊步骤标识，表示搜索失败
            "messages": [AIMessage(content="❌ 搜索遇到问题，我将基于已有知识为您回答")]  # 用户提示
        }

def generate_answer_node(state: SearchState) -> SearchState:
    """步骤3：基于搜索结果生成最终答案"""
    
    # 检查是否有搜索结果 - 如果上一步搜索失败
    if state["step"] == "search_failed":
        # 如果搜索失败，基于LLM的已有知识回答
        fallback_prompt = f"""搜索API暂时不可用，请基于您的知识回答用户的问题：

用户问题：{state['user_query']}

请提供一个有用的回答，并说明这是基于已有知识的回答。"""
        
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
        
        return {
            "final_answer": response.content,  # 基于知识的回答
            "step": "completed",  # 标记完成
            "messages": [AIMessage(content=response.content)]  # 添加最终回答消息
        }
    
    # 基于搜索结果生成答案 - 正常流程
    answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案：

用户问题：{state['user_query']}

搜索结果：
{state['search_results']}

请要求：
1. 综合搜索结果，提供准确、有用的回答
2. 如果是技术问题，提供具体的解决方案或代码
3. 引用重要信息的来源
4. 回答要结构清晰、易于理解
5. 如果搜索结果不够完整，请说明并提供补充建议"""

    # 调用语言模型生成最终答案
    response = llm.invoke([SystemMessage(content=answer_prompt)])
    
    return {
        "final_answer": response.content,  # 最终生成的答案
        "step": "completed",  # 标记工作流完成
        "messages": [AIMessage(content=response.content)]  # 添加最终回答消息
    }

# 构建搜索工作流
def create_search_assistant():
    # 创建状态图，指定状态结构
    workflow = StateGraph(SearchState)
    
    # 添加三个处理节点
    workflow.add_node("understand", understand_query_node)  # 理解查询节点
    workflow.add_node("search", tavily_search_node)         # 搜索节点
    workflow.add_node("answer", generate_answer_node)       # 生成答案节点
    
    # 设置线性流程 - 定义节点执行顺序
    workflow.add_edge(START, "understand")     # 从开始到理解节点
    workflow.add_edge("understand", "search")  # 从理解到搜索节点
    workflow.add_edge("search", "answer")      # 从搜索到答案节点
    workflow.add_edge("answer", END)           # 从答案节点到结束
    
    # 编译图，添加内存检查点保存器
    memory = InMemorySaver()  # 创建内存检查点保存器，用于保存状态
    app = workflow.compile(checkpointer=memory)  # 编译成可执行应用
    
    return app

async def main():
    """主函数：运行智能搜索助手"""
    
    # 检查API密钥 - 确保必要的环境变量已配置
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 错误：请在.env文件中配置TAVILY_API_KEY")
        return
    
    # 创建搜索助手应用
    app = create_search_assistant()
    
    # 打印欢迎信息
    print("🔍 智能搜索助手启动！")
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持各种问题：新闻、技术、知识问答等")
    print("(输入 'quit' 退出)\n")
    
    session_count = 0  # 会话计数器，用于生成唯一的会话ID
    
    while True:
        # 获取用户输入
        user_input = input("🤔 您想了解什么: ").strip()
        
        # 检查退出命令
        if user_input.lower() in ['quit', 'q', '退出', 'exit']:
            print("感谢使用！再见！👋")
            break
        
        # 跳过空输入
        if not user_input:
            continue
        
        # 递增会话计数，生成唯一会话ID
        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}
        
        # 初始化状态 - 设置工作流的初始状态
        initial_state = {
            "messages": [HumanMessage(content=user_input)],  # 用户消息
            "user_query": "",  # 初始为空，将由理解节点填充
            "search_query": "",  # 初始为空，将由理解节点填充
            "search_results": "",  # 初始为空，将由搜索节点填充
            "final_answer": "",  # 初始为空，将由答案节点填充
            "step": "start"  # 初始步骤
        }
        
        try:
            print("\n" + "="*60)
            
            # 执行工作流 - 使用异步流式执行，实时显示进度
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    # 检查每个节点的输出，提取并显示消息
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        if isinstance(latest_message, AIMessage):
                            # 根据节点名称显示不同的状态信息
                            if node_name == "understand":
                                print(f"🧠 理解阶段: {latest_message.content}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message.content}")
                            elif node_name == "answer":
                                print(f"\n💡 最终回答:\n{latest_message.content}")
            
            print("\n" + "="*60 + "\n")
        
        except Exception as e:
            # 异常处理 - 捕获并显示错误
            print(f"❌ 发生错误: {e}")
            print("请重新输入您的问题。\n")

if __name__ == "__main__":
    # 运行主函数 - 使用asyncio运行异步主函数
    asyncio.run(main())