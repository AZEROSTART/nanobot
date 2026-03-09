"""Agent loop: the core processing engine.

该模块驱动消息接收、构建上下文、调用 LLM、执行工具调用并发送回复。
注释旨在逐行解释 Python 习惯用法与代码意图，便于 Go 开发者理解。
"""

from __future__ import annotations  # 提前启用未来版本的类型注解行为（允许使用 X | Y 语法）

import asyncio  # 异步库：事件循环、任务、锁等
import json  # 序列化/反序列化 JSON
import re  # 正则，用于文本处理
import weakref  # 弱引用，用于缓存不会阻止对象被回收的引用
from contextlib import AsyncExitStack  # 异步上下文管理栈，用于管理异步资源
from pathlib import Path  # 面向对象的路径操作
from typing import TYPE_CHECKING, Any, Awaitable, Callable  # 类型注解辅助

from loguru import logger  # 日志库（第三方），类似 Go 的 log

from nanobot.agent.context import ContextBuilder  # 构建 LLM 上下文
from nanobot.agent.memory import MemoryStore  # 长期记忆读写封装
from nanobot.agent.subagent import SubagentManager  # 子代理管理器（spawn 子任务）
from nanobot.agent.tools.cron import CronTool  # 定时任务工具
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool  # 文件工具
from nanobot.agent.tools.message import MessageTool  # 发消息工具
from nanobot.agent.tools.registry import ToolRegistry  # 注册和查找工具的注册表
from nanobot.agent.tools.shell import ExecTool  # 执行 shell 命令的工具
from nanobot.agent.tools.spawn import SpawnTool  # 启动子 agent 的工具
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool  # 网络抓取/搜索工具
from nanobot.bus.events import InboundMessage, OutboundMessage  # 总线事件的数据结构
from nanobot.bus.queue import MessageBus  # 消息总线接口
from nanobot.providers.base import LLMProvider  # LLM 提供者抽象
from nanobot.session.manager import Session, SessionManager  # 会话及其管理器

if TYPE_CHECKING:
    # 仅在类型检查时导入以避免运行时循环或开销，类似 Go 的 build tags
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 500

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        # 延迟导入类型以避免循环引用问题
        from nanobot.config.schema import ExecToolConfig

        # 将传入参数保存到实例属性中
        self.bus = bus  # 消息总线实例，用于接收/发送
        self.channels_config = channels_config  # 可选的通道配置
        self.provider = provider  # LLM 提供者实例
        self.workspace = workspace  # 工作目录路径
        # 如果未指定模型，使用 provider 提供的默认模型
        self.model = model or provider.get_default_model()
        # 控制最大迭代次数、温度、token 限制等 LLM 参数
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        # exec_config 可能为 None，使用默认 ExecToolConfig()
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        # 构建上下文、会话管理器、工具注册表与子代理管理器
        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=reasoning_effort,
            brave_api_key=brave_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        # 控制循环与 MCP（消息/协作协议）状态
        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        # 追踪正在执行归档(consolidation)的 session keys
        self._consolidating: set[str] = set()
        # 保存活跃的归档任务引用，防止被垃圾回收
        self._consolidation_tasks: set[asyncio.Task] = set()
        # 每个会话的归档锁（弱引用字典以允许锁被回收）
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        # 活动任务映射：session_key -> [asyncio.Task,...]
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        # 全局处理锁，确保同一时刻只有一个消息在处理（避免竞态）
        self._processing_lock = asyncio.Lock()
        # 注册默认工具（读写文件、执行命令、网络等）
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # 如果限制为仅工作区，则 allowed_dir 指向工作区路径，否则为 None（表示不限制）
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        # 注册一组文件系统相关工具，传入 workspace 与 allowed_dir
        # 注册的工具
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))

        # 注册执行命令工具，传入工作目录、超时、路径追加等配置
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))

        # 注册网络搜索与抓取工具（可能需要 API key 或代理）
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))

        # 注册消息发送工具，MessageTool 需要一个回调用于最终发送消息到总线
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))

        # 注册 spawn 工具以支持启动子 agent（subagent）
        self.tools.register(SpawnTool(manager=self.subagents))

        # 如果有 cron 服务，则注册定时任务工具
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        # 如果已经连接或正在连接，或没有配置 MCP 服务器，则跳过
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        # 标记为正在连接，避免并发重复连接
        self._mcp_connecting = True
        # 延迟导入 mcp 连接方法，避免模块级循环依赖
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            # 使用 AsyncExitStack 管理异步资源的打开/关闭
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            # 调用工具函数向 MCP 服务器发起连接，并把 tools 与 exit stack 传入
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            # 连接失败时记录错误，并尝试清理已创建的 stack
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            # 无论成功或失败，都将 connecting 标志重置
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        # 某些工具需要知道当前消息的 routing 信息（channel/chat_id/message_id），逐个设置
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                # hasattr 检查对象是否实现 set_context 方法
                if hasattr(tool, "set_context"):
                    # 为 message 工具传入 message_id，其余工具不需要该参数
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        # 如果文本为空直接返回 None
        if not text:
            return None
        # 使用正则去掉 <think>...</think> 区块，并去除首尾空白，若结果为空则返回 None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        # 将工具调用列表格式化为短提示，便于在进度信息中展示
        def _fmt(tc):
            # 一些工具调用将 arguments 放在列表中（取第一个元素），否则直接使用 dict
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            # 尝试从参数字典中取第一个值作为展示内容
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                # 如果无法提取字符串参数，则只返回工具名
                return tc.name
            # 截断过长的参数以保持提示简洁
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'

        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        # 主循环：模型可能返回需要工具调用的响应，循环直到没有工具调用或达到最大迭代次数
        while iteration < self.max_iterations:
            iteration += 1

            # 调用 LLM 提供者，传入当前的消息列表和工具定义
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )

            # 如果模型发起了工具调用（function/tool calls）
            if response.has_tool_calls:
                # 如果提供了进度回调，则先发送模型的可读内容（去掉 <think>）
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    # 再发送一个简短的工具调用提示，例如 web_search("query")
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                # 将模型返回的工具调用转换为内部的 dict 结构，便于上下文记录
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]

                # 把 assistant 的消息（包含工具调用元数据）追加到 messages
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                # 依次执行模型请求的每个工具调用，并将工具结果添加回 messages
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    # execute 异步调用注册的工具并返回结果字符串
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # 模型没有工具调用，处理最终文本返回
                clean = self._strip_think(response.content)
                # 如果模型返回错误，记录日志并返回错误信息，而不把错误持久化到会话
                # （否则可能污染上下文并导致永久性的 400 错误循环）
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                # 将 assistant 的最终响应追加到消息，并设置 final_content 以结束循环
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        # 启动主循环前先连接 MCP（一次性延迟连接）
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        # 主循环：持续消费总线上的入站消息
        while self._running:
            try:
                # 使用 timeout 确保可以定期检查 _running 标志并响应 /stop
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # 简单的内置命令：/stop 直接取消当前 session 的任务
            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                # 将消息交给 _dispatch 异步处理，并记录 task 以便可能取消
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                # 添加完成回调以自动从活动任务列表移除已完成的 task
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        # 从活动任务字典中移除并取消对应 session 的所有任务
        tasks = self._active_tasks.pop(msg.session_key, [])
        # 对未完成的任务调用 cancel() 并统计被取消的数量
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                # 等待任务结束以确保资源清理
                await t
            except (asyncio.CancelledError, Exception):
                pass
        # 取消子代理相关的任务并统计
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        # 将结果通过总线发送回客户端
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        # 使用全局锁确保消息处理的串行化（避免并发修改 session 等）
        async with self._processing_lock:
            try:
                # 处理单条消息并返回 OutboundMessage（或 None）
                response = await self._process_message(msg)
                if response is not None:
                    # 将响应发布到总线
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    # 对 CLI 通道，若没有响应也发送空回复以保持交互一致
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                # 任务被取消时记录并向上抛以保持取消语义
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                # 捕获所有异常并报告到总线，以免崩溃主循环
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        # 关闭 MCP 的 AsyncExitStack（如果存在），忽略某些已知可忽略的异常
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        # 设置标志以停止主循环
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # 处理系统消息（channel == "system"）：system 消息的 chat_id 中包含真实来源 channel:chat_id
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            # 构造 session key 并获取会话对象
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            # 为工具设置路由上下文
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            # 获取历史（用于上下文），构建 messages 列表并运行 agent loop
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            # 将新生成的消息保存到会话并写盘
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        # 日志记录消息预览（避免过长的完整文本）
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        # 获取或创建会话（允许外部传入 session_key 以复用会话）
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # 支持简单的斜杠命令：/new、/help
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # 为当前 session 获取或创建归档锁，并标记正在归档以避免重复
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())
            self._consolidating.add(session.key)
            try:
                async with lock:
                    # snapshot 未归档的消息并尝试归档（archive_all=True）
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                # 取消正在归档标记
                self._consolidating.discard(session.key)

            # 清空会话并保存
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/stop — Stop the current task\n/help — Show available commands")

        # 如果未归档消息数超过 memory_window，则异步触发归档
        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    # 无论成功或失败，都移除正在归档的标记并从任务集合中删除当前任务
                    self._consolidating.discard(session.key)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            # 将归档任务放到后台执行，不阻塞当前消息的处理
            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        # 为工具设置当前消息上下文并在 MessageTool 上开启一个新回合（如果存在）
        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        # 构建供 LLM 使用的历史 + 当前消息上下文
        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            # 通过总线发送进度更新，携带 _progress 和可选的 _tool_hint 元数据
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        # 执行 agent 循环主逻辑，传入用于进度上报的回调
        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        # 如果没有返回内容，提供一个友好的默认消息
        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # 将新产生的消息追加到 session 并保存
        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        # 如果 MessageTool 已在本回合发送了消息（例如代理直接发送），则不要再通过总线重复发送
        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        # 记录发送内容的预览并返回 OutboundMessage
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        # 将 messages 中从索引 skip 开始的新消息追加到 session
        from datetime import datetime
        for m in messages[skip:]:
            # 复制 dict 避免修改原始结构
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            # 跳过空的 assistant 消息（可能来自模型的中间步骤）
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            # 如果工具返回结果过长，则截断以避免膨胀会话文件
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                # 如果用户消息是运行时上下文元数据，则不存入会话历史
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    continue
                # 如果用户消息包含图片数据（列表形式），则替换 data:image/ 内联内容以节省空间
                if isinstance(content, list):
                    entry["content"] = [
                        {"type": "text", "text": "[image]"} if (
                            c.get("type") == "image_url"
                            and c.get("image_url", {}).get("url", "").startswith("data:image/")
                        ) else c for c in content
                    ]
            # 确保每条消息都有时间戳
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        # 将合并任务委托给 MemoryStore 实例执行，返回 bool 表示是否成功
        return await MemoryStore(self.workspace).consolidate(
            session, self.provider, self.model,
            archive_all=archive_all, memory_window=self.memory_window,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        # 供 CLI 或定时任务直接调用：构造 InboundMessage 并调用 _process_message
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
