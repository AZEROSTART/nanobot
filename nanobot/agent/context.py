"""Context builder for assembling agent prompts.

此模块负责为 LLM 调用构建系统提示和消息列表。
注释力求对每一段代码给出简短解释，帮助来自 Go 背景的开发者理解 Python 习惯用法。
"""

import base64  # 用于将二进制文件（图片）编码为 base64 文本
import mimetypes  # 用于猜测文件的 MIME 类型
import platform  # 获取运行时平台信息（如 macOS/Windows/Linux）
import time  # 提供时区等运行时信息
from datetime import datetime  # 精确到日期时间的工具
from pathlib import Path  # 面向对象的路径操作（跨平台）
from typing import Any  # 类型注解中使用的通用类型

from nanobot.agent.memory import MemoryStore  # 长期记忆读写封装
from nanobot.agent.skills import SkillsLoader  # 技能加载器（从 SKILL.md 读取）


class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent.

    这里封装了构建 system prompt（系统提示）和对话消息列表的逻辑。
    属性说明：
    - `workspace`: 工作目录（Path 对象），所有文件相对于此目录。
    - `memory`: MemoryStore 实例，用于读取写入长期记忆文件（MEMORY.md/HISTORY.md）。
    - `skills`: SkillsLoader 实例，用于列举和加载技能文件（SKILL.md）。
    """

    # 启动时会尝试加载的引导文件列表（如果存在则包含其内容）
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]

    # 这是注入到用户消息前面的运行时上下文标签（仅元数据，不是指令）
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

    def __init__(self, workspace: Path):
        # 将传入的 Path 保存到实例中，后续方法使用 self.workspace
        self.workspace = workspace
        # MemoryStore 用于从磁盘读取/写入长期记忆
        self.memory = MemoryStore(workspace)
        # SkillsLoader 用于查找并加载 SKILL.md 文件
        self.skills = SkillsLoader(workspace)

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and skills.

        该方法拼接多个部分形成 LLM 的 system prompt：
        1. 身份（identity）和运行时信息
        2. 可选的引导文件（workspace 下的 AGENTS.md 等）
        3. 长期记忆（MEMORY.md）
        4. 始终激活的技能（always=true 的 SKILL.md）
        5. 技能汇总（用于渐进加载提示）
        最终返回一个字符串，调用方会将其作为 system 消息传给 LLM。
        """

        # parts 列表用于按顺序保存不同段落，最后用分隔符连接
        parts = [self._get_identity()]

        # 尝试加载工作区中的引导文件（例如 AGENTS.md）
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        # 将长期记忆插入系统提示，若存在则添加
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        # 加载被标记为 always 的技能（满足要求时始终激活）
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # 构造技能汇总（XML），便于在 system prompt 中报告可用技能
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")

        # 用显式分隔符连接各部分，形成最终系统提示文本
        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """Get the core identity section.

        返回一个字符串，描述机器人身份、运行时信息、工作区路径以及若干使用指南。
        这部分被放在系统提示的最前面，指导 LLM 的总体行为。
        """

        # 将 workspace 路径标准化并展开 ~ 等符号
        workspace_path = str(self.workspace.expanduser().resolve())

        # platform.system() 返回操作系统名称 (Darwin/Windows/Linux)
        system = platform.system()
        # 构造运行时描述字符串：平台 + 机器架构 + Python 版本
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        # 使用多行 f-string 返回详细的身份与指南内容
        return f"""# nanobot 🐈

You are nanobot, a helpful AI assistant.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md (write important facts here)
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable). Each entry starts with [YYYY-MM-DD HH:MM].
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

## nanobot Guidelines
- State intent before tool calls, but NEVER predict or claim results before receiving them.
- Before modifying a file, read it first. Do not assume files or directories exist.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.
- Ask for clarification when the request is ambiguous.

Reply directly with text for conversations. Only use the 'message' tool to send to a specific chat channel."""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """Build untrusted runtime metadata block for injection before the user message.

        这是一个不可信的元数据块（runtime metadata），仅描述时间、通道和聊天 ID。
        注意：该信息仅供参考，不应作为 LLM 的指令来源（因此在标签中有说明）。
        """

        # 当前时间，格式化为可读字符串
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        # 获取时区缩写，若不可用则回退到 UTC
        tz = time.strftime("%Z") or "UTC"
        lines = [f"Current Time: {now} ({tz})"]
        # 如果提供了 channel 和 chat_id，则将它们加入元数据行
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        # 将标签和所有元数据行拼接为最终字符串返回
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace.

        遍历 BOOTSTRAP_FILES，若对应文件存在则读取其文本并作为一个段落返回。
        返回值为拼接后的字符串或空字符串。
        """

        parts = []

        for filename in self.BOOTSTRAP_FILES:
            # 使用 Path 的 / 操作符拼接路径（Path 对象重载了 /）
            file_path = self.workspace / filename
            if file_path.exists():
                # 读取文件文本（默认 utf-8 编码）并添加到 parts
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        # 将存在的引导文件段落以空行分隔连接；若无则返回空字符串
        return "\n\n".join(parts) if parts else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the complete message list for an LLM call.

        返回一个消息列表（list of dict），顺序为：system prompt、历史消息、运行时元数据、用户消息（可能包含图片）。
        注意：这里使用了 Python 的序列解包语法 `*history` 将历史消息插入到列表中。
        """

        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},
            *history,  # 将历史消息按原样插入到系统提示之后
            {"role": "user", "content": self._build_runtime_context(channel, chat_id)},
            {"role": "user", "content": self._build_user_content(current_message, media)},
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images.

        如果没有 media，直接返回纯文本字符串。
        若有图片路径，则把本地图片读为 bytes，再编码为 data URL（data:{mime};base64,...），
        返回一个包含 image_url 对象和文本块的列表（部分 LLM 提供者支持此类富结构）。
        返回类型：要么是字符串，要么是 list[dict]。
        """

        # 无媒体时直接返回纯文本
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)  # 把传入路径转换为 Path 对象以便使用 is_file/read_bytes 等方法
            mime, _ = mimetypes.guess_type(path)  # 猜测 MIME 类型
            # 仅处理存在且为图片的文件
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            # 读取文件内容并进行 base64 编码，再 decode 为字符串
            b64 = base64.b64encode(p.read_bytes()).decode()
            # 将图片封装成 provider 希望的 image_url 结构
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        # 如果没有有效图片，退回为纯文本
        if not images:
            return text
        # 返回图片列表并在末尾附上文本块（部分 LLM 接受富结构消息）
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str,
    ) -> list[dict[str, Any]]:
        """Add a tool result to the message list.

        工具的返回结果被包装为 role="tool" 的消息追加到 messages 中，
        以便后续的 LLM 调用能看到工具执行的输出。
        """

        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Add an assistant message to the message list.

        此方法用于将 assistant（模型生成）的消息追加到 messages 中。
        支持附带工具调用元数据、推理内容（可能是对模型内部思路的结构化记录）以及思考块。
        返回更新后的 messages 列表（与 Python 的列表是可变对象，调用者通常无需使用返回值）。
        """

        msg: dict[str, Any] = {"role": "assistant", "content": content}
        # 可选字段：tool_calls（模型发起的工具调用列表）
        if tool_calls:
            msg["tool_calls"] = tool_calls
        # reasoning_content 用于保存可选的推理文本
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        # thinking_blocks 可能包含模型分段的思考状态
        if thinking_blocks:
            msg["thinking_blocks"] = thinking_blocks
        messages.append(msg)
        return messages
