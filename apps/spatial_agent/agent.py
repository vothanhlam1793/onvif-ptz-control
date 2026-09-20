"""LangGraph Spatial Memory Agent with Turn Integrity, Tool Calling, and Persistent Memory."""

from __future__ import annotations

import json
import logging
import os
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from apps.spatial_agent.prompts import SPATIAL_AGENT_SYSTEM_PROMPT
from apps.spatial_agent.tools import get_spatial_agent_tools
from apps.spatial_agent.db import save_message, load_session_messages, ensure_session

load_dotenv()
logger = logging.getLogger(__name__)


def create_agent_model(temperature: float = 0.2) -> ChatOpenAI:
    """Create ChatOpenAI model configured from .env (Gemini 3.7 / 2.5 Flash via 9Router)."""
    base_url = os.getenv("NINEROUTER_BASE_URL", "https://9router.camerangochoang.com/v1")
    api_key = os.getenv("NINEROUTER_API_KEY", "")
    model = os.getenv("VLM_MODEL", "ag/gemini-3.7-flash-high")
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=2000,
        timeout=60.0,
    )


def compact_messages(
    messages: list[AnyMessage],
    max_user_turns: int = 5,
    max_tool_chars: int = 1500,
) -> list[AnyMessage]:
    """Strictly sanitize message sequence to guarantee Gemini / LLM turn integrity (CRETA Standard)."""
    if not messages:
        return []

    # 1. Truncate lengthy tool contents
    sanitized: list[AnyMessage] = []
    for m in messages:
        if isinstance(m, ToolMessage) and isinstance(m.content, str) and len(m.content) > max_tool_chars:
            sanitized.append(
                m.model_copy(update={"content": m.content[:max_tool_chars] + "\n[Đã rút gọn kết quả tool]"})
            )
        else:
            sanitized.append(m)

    # 2. Slice to latest N human turns
    human_indices = [i for i, m in enumerate(sanitized) if isinstance(m, HumanMessage)]
    if human_indices and len(human_indices) > max_user_turns:
        sanitized = sanitized[human_indices[-max_user_turns] :]
    elif not human_indices and len(sanitized) > 10:
        sanitized = sanitized[-10:]

    # 3. Clean and validate turn by turn
    cleaned: list[AnyMessage] = []
    i = 0
    while i < len(sanitized):
        msg = sanitized[i]

        if isinstance(msg, HumanMessage):
            # If previous message was a ToolMessage without AI conclusion, close it
            if cleaned and isinstance(cleaned[-1], ToolMessage):
                cleaned.append(AIMessage(content="Đã xử lý xong dữ liệu tool."))
            
            # If previous message was already HumanMessage, combine text
            if cleaned and isinstance(cleaned[-1], HumanMessage):
                cleaned[-1] = HumanMessage(content=f"{cleaned[-1].content}\n{msg.content}")
            else:
                cleaned.append(msg)
            i += 1

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                expected_ids = {tc.get("id") for tc in msg.tool_calls if isinstance(tc, dict) and tc.get("id")}
                found_tools: list[AnyMessage] = []
                j = i + 1
                while j < len(sanitized) and isinstance(sanitized[j], ToolMessage):
                    if sanitized[j].tool_call_id in expected_ids:
                        found_tools.append(sanitized[j])
                    j += 1
                
                # Only keep AIMessage(tool_calls) if all tool responses are present
                if found_tools and len(found_tools) == len(expected_ids):
                    cleaned.append(msg)
                    cleaned.extend(found_tools)
                    if j < len(sanitized) and isinstance(sanitized[j], AIMessage) and not sanitized[j].tool_calls:
                        cleaned.append(sanitized[j])
                        i = j + 1
                    else:
                        i = j
                else:
                    if msg.content:
                        cleaned.append(AIMessage(content=msg.content))
                    i += 1
            else:
                if msg.content:
                    cleaned.append(msg)
                i += 1

        elif isinstance(msg, ToolMessage):
            # Orphaned tool message without preceding AIMessage -> ignore
            i += 1
        else:
            cleaned.append(msg)
            i += 1

    if cleaned and isinstance(cleaned[-1], ToolMessage):
        cleaned.append(AIMessage(content="Đã hoàn tất thao tác camera."))

    while cleaned and not isinstance(cleaned[0], HumanMessage):
        cleaned.pop(0)

    while cleaned and isinstance(cleaned[-1], AIMessage):
        cleaned.pop(-1)

    return cleaned


class SpatialAgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]


def build_spatial_agent(tools: list | None = None, checkpointer=None):
    """Build the compiled LangGraph workflow for Spatial Memory PTZ Agent."""
    if tools is None:
        tools = get_spatial_agent_tools()

    llm = create_agent_model(temperature=0.2)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def chatbot_node(state: SpatialAgentState) -> dict:
        raw_messages = list(state["messages"])
        compacted = compact_messages(raw_messages)
        messages_with_system = [SystemMessage(content=SPATIAL_AGENT_SYSTEM_PROMPT)] + compacted
        response = llm_with_tools.invoke(messages_with_system)
        return {"messages": [response]}

    def route_tools(state: SpatialAgentState) -> str:
        messages = state["messages"]
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(SpatialAgentState)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
    workflow.add_edge("tools", "chatbot")

    return workflow.compile(checkpointer=checkpointer)


async def execute_spatial_turn(
    agent_app,
    session_id: str,
    user_text: str,
    channel: str = "cli",
    user_id: str = "guest",
    user_name: str = "Khách",
) -> dict:
    """Execute a complete LangGraph agent turn with persistent SQLite memory."""
    ensure_session(session_id, channel=channel, user_id=user_id, user_name=user_name, title=user_text[:30])

    # 1. Save incoming human message to DB
    save_message(session_id=session_id, role="user", content=user_text)

    # 2. Load prior conversation history from SQLite and compact
    raw_history = load_session_messages(session_id, limit=20)
    history_messages: list[AnyMessage] = []
    for r in raw_history:
        role = r["role"]
        content = r["content"]
        if role == "user":
            history_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            tc = None
            if r.get("tool_name") and r.get("tool_args_json"):
                try:
                    args = json.loads(r["tool_args_json"])
                    tc = [{"name": r["tool_name"], "args": args, "id": r.get("tool_call_id") or "call_1"}]
                except Exception:
                    pass
            history_messages.append(AIMessage(content=content, tool_calls=tc or []))
        elif role == "tool":
            history_messages.append(ToolMessage(content=content, tool_call_id=r.get("tool_call_id") or "call_1", name=r.get("tool_name") or "tool"))

    compacted = compact_messages(history_messages)

    # 3. Run LangGraph
    response = await agent_app.ainvoke({"messages": compacted})

    # 4. Extract newly generated messages
    all_output_msgs = response.get("messages", [])
    num_input_msgs = len(compacted)
    new_msgs = all_output_msgs[num_input_msgs:] if len(all_output_msgs) > num_input_msgs else all_output_msgs

    assistant_msg = ""
    tool_actions = []

    for msg in new_msgs:
        if isinstance(msg, AIMessage):
            if msg.content:
                assistant_msg = str(msg.content)
            
            tool_args_str = None
            tool_name = None
            tool_call_id = None
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                tool_name = tc.get("name")
                tool_args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)
                tool_call_id = tc.get("id")
            
            if msg.content or msg.tool_calls:
                save_message(
                    session_id=session_id,
                    role="assistant",
                    content=str(msg.content or ""),
                    tool_name=tool_name,
                    tool_args_json=tool_args_str,
                    tool_call_id=tool_call_id,
                )

        elif isinstance(msg, ToolMessage):
            t_name = getattr(msg, "name", "tool")
            t_content = str(msg.content)
            tool_actions.append({"name": t_name, "content": t_content})
            save_message(
                session_id=session_id,
                role="tool",
                content=t_content,
                tool_name=t_name,
                tool_call_id=msg.tool_call_id,
            )

    return {
        "session_id": session_id,
        "reply": assistant_msg or "Camera đã hoàn tất thao tác định vị không gian ✨",
        "tool_actions": tool_actions,
    }
