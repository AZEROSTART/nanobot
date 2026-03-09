"""CLI commands for nanobot.

整体架构类比 Go 的 cobra CLI 框架：
- typer.Typer()  ≈ Go cobra.Command{} — 定义命令和子命令
- asyncio       ≈ Go goroutine + channel — 并发模型
- MessageBus    ≈ Go channel — 协程间消息传递
- @app.command() ≈ Go cmd.AddCommand() — 注册子命令
"""

import asyncio          # ≈ Go 的 goroutine 运行时，提供事件循环和协程调度
import os
import select           # ≈ Go 的 select{}，多路 IO 复用
import signal           # ≈ Go 的 os/signal 包，处理系统信号(如 Ctrl+C)
import sys
from pathlib import Path  # ≈ Go 的 path/filepath 包

import typer              # ≈ Go 的 cobra 库，CLI 命令框架
from prompt_toolkit import PromptSession       # ≈ Go 的 liner/readline 库，交互式输入
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory  # 命令历史持久化，类似 shell 的 .bash_history
from prompt_toolkit.patch_stdout import patch_stdout  # 防止异步输出和用户输入互相干扰
from rich.console import Console    # ≈ Go 的 color/lipgloss 库，终端美化输出
from rich.markdown import Markdown  # 将 Markdown 渲染到终端
from rich.table import Table        # 终端中画表格
from rich.text import Text

from nanobot import __logo__, __version__
from nanobot.config.schema import Config
from nanobot.utils.helpers import sync_workspace_templates

# ============================================================================
# 应用初始化 —— 类比 Go: var rootCmd = &cobra.Command{Use: "nanobot"}
# ============================================================================
app = typer.Typer(
    name="nanobot",
    help=f"{__logo__} nanobot - Personal AI Assistant",
    no_args_is_help=True,  # 无参数时显示帮助，≈ cobra 的 RunE 返回 help
)

console = Console()  # 全局终端输出器，≈ Go 中全局的 fmt/log writer
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}  # 退出指令集合，≈ Go map[string]bool

# ---------------------------------------------------------------------------
# CLI 输入处理——类比 Go 里用 liner 库做交互式 readline
# _PROMPT_SESSION 是全局单例，≈ Go 中的 var liner *liner.State
# _SAVED_TERM_ATTRS 保存终端原始配置，类似 Go 中 defer term.Restore(fd, oldState)
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None  # 全局的交互式输入会话单例
_SAVED_TERM_ATTRS = None  # 终端原始配置快照，程序退出时恢复


def _flush_pending_tty_input() -> None:
    """清空终端输入缓冲区中残留的按键。

    类比 Go：在 agent 生成回复期间用户可能误按了键盘，
    就像 Go 中读完 channel 后需要 drain 掉多余的消息：
        for len(ch) > 0 { <-ch }

    优先用 termios.tcflush（系统级清空），
    降级方案用 select 非阻塞轮询读取丢弃。
    """
    try:
        fd = sys.stdin.fileno()  # 获取标准输入的文件描述符，≈ Go os.Stdin.Fd()
        if not os.isatty(fd):   # 不是真实终端则跳过(如管道输入)
            return
    except Exception:
        return

    # 方案一：用 termios 系统调用直接刷掉输入缓冲区（最快）
    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    # 方案二：select 非阻塞轮询，≈ Go 的 select + default 非阻塞读取
    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)  # timeout=0 非阻塞
            if not ready:
                break
            if not os.read(fd, 4096):  # 读取并丢弃
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """恢复终端到程序启动前的原始状态。

    类比 Go：
        oldState, _ := term.MakeRaw(fd)
        defer term.Restore(fd, oldState)  // <-- 就是这个 defer

    prompt_toolkit 会修改终端模式(关闭回显等)，退出时必须恢复，
    否则终端会变得不可用(看不到输入的字符)。
    """
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    """初始化交互式输入会话（带历史记录持久化）。

    类比 Go：
        line := liner.NewLiner()
        line.SetCtrlCAborts(true)
        f, _ := os.Open("~/.nanobot/history/cli_history")
        line.ReadHistory(f)

    这个函数只在交互模式启动时调用一次，创建全局单例。
    """
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    # 保存终端原始状态快照，退出时用 _restore_terminal() 恢复
    # ≈ Go: oldState, _ = term.GetState(fd)
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    # 命令历史持久化到文件，下次启动可用 ↑↓ 翻阅
    history_file = Path.home() / ".nanobot" / "history" / "cli_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),  # ≈ Go liner.ReadHistory()
        enable_open_in_editor=False,
        multiline=False,   # 按 Enter 立即提交，不等多行
    )


def _print_agent_response(response: str, render_markdown: bool) -> None:
    """格式化打印 agent 的回复到终端。

    类比 Go：fmt.Printf("\n🤖 nanobot\n%s\n", glamour.Render(content))
    render_markdown=True 时将 Markdown 渲染为彩色终端输出（标题、代码块等）。
    """
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print()
    console.print(f"[cyan]{__logo__} nanobot[/cyan]")
    console.print(body)
    console.print()


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


async def _read_interactive_input_async() -> str:
    """异步读取用户输入（支持粘贴、历史翻阅）。

    类比 Go：
        text, err := liner.Prompt("You: ")
        // 但这里是 async 版本，不会阻塞事件循环

    patch_stdout() 确保 agent 的异步输出不会破坏用户正在编辑的输入行，
    类似 Go 中用 mutex 保护终端输出不和输入冲突。

    prompt_async 是 prompt_toolkit 的异步版本，
    ≈ Go 中把 liner.Prompt 放在单独的 goroutine 里用 channel 返回结果。
    """
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():  # 保护输入行不被异步输出打乱
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),  # 蓝色加粗提示符
            )
    except EOFError as exc:  # Ctrl+D 触发 EOF
        raise KeyboardInterrupt from exc



# 版本回调 —— 类比 Go cobra 中的 PersistentPreRun
# 当用户执行 nanobot --version 时就会触发这个回调，打印版本后直接退出
def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} nanobot v{__version__}")
        raise typer.Exit()  # ≈ Go 的 os.Exit(0)


# @app.callback() ≈ Go cobra 的 rootCmd.PersistentPreRun
# 这是所有子命令执行前的全局钩子
@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
        # is_eager=True 表示优先解析这个参数，不等其他参数
    ),
):
    """nanobot - Personal AI Assistant."""
    pass


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()  # 注册为子命令：nanobot onboard
def onboard():
    """初始化配置和工作空间。

    类比 Go：
        var initCmd = &cobra.Command{Use: "init", RunE: func(cmd, args) { ... }}

    工作流程：
    1. 检查配置文件 ~/.nanobot/config.json 是否存在
       - 存在：询问用户是覆盖还是刷新(保留已有值)
       - 不存在：创建默认配置
    2. 创建工作空间目录
    3. 同步模板文件到工作空间
    """
    from nanobot.config.loader import get_config_path, load_config, save_config
    from nanobot.config.schema import Config
    from nanobot.utils.helpers import get_workspace_path

    config_path = get_config_path()

    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        console.print("  [bold]y[/bold] = overwrite with defaults (existing values will be lost)")
        console.print("  [bold]N[/bold] = refresh config, keeping existing values and adding new fields")
        if typer.confirm("Overwrite?"):
            config = Config()
            save_config(config)
            console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
        else:
            config = load_config()
            save_config(config)
            console.print(f"[green]✓[/green] Config refreshed at {config_path} (existing values preserved)")
    else:
        save_config(Config())
        console.print(f"[green]✓[/green] Created config at {config_path}")

    # Create workspace
    workspace = get_workspace_path()

    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created workspace at {workspace}")

    sync_workspace_templates(workspace)

    console.print(f"\n{__logo__} nanobot is ready!")
    console.print("\nNext steps:")
    console.print("  1. Add your API key to [cyan]~/.nanobot/config.json[/cyan]")
    console.print("     Get one at: https://openrouter.ai/keys")
    console.print("  2. Chat: [cyan]nanobot agent -m \"Hello!\"[/cyan]")
    console.print("\n[dim]Want Telegram/WhatsApp? See: https://github.com/HKUDS/nanobot#-chat-apps[/dim]")





def _make_provider(config: Config):
    """工厂函数：根据配置创建合适的 LLM 提供者实例。

    类比 Go 的接口 + 工厂模式：
        type Provider interface {
            Complete(ctx, messages) (string, error)
        }
        func NewProvider(config Config) Provider {
            switch config.ProviderName {
            case "openai_codex": return &CodexProvider{}
            case "custom":      return &CustomProvider{}
            default:            return &LiteLLMProvider{}
            }
        }

    支持三种提供者：
    1. OpenAI Codex（OAuth 认证）
    2. Custom（自部署的 OpenAI 兼容服务，跳过 LiteLLM）
    3. LiteLLM（通用代理，支持多家模型如 OpenRouter/Anthropic/Google 等）
    """
    from nanobot.providers.custom_provider import CustomProvider
    from nanobot.providers.litellm_provider import LiteLLMProvider
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)

    # OpenAI Codex (OAuth)
    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        return OpenAICodexProvider(default_model=model)

    # Custom: direct OpenAI-compatible endpoint, bypasses LiteLLM
    if provider_name == "custom":
        return CustomProvider(
            api_key=p.api_key if p else "no-key",
            api_base=config.get_api_base(model) or "http://localhost:8000/v1",
            default_model=model,
        )

    from nanobot.providers.registry import find_by_name
    spec = find_by_name(provider_name)
    if not model.startswith("bedrock/") and not (p and p.api_key) and not (spec and spec.is_oauth):
        console.print("[red]Error: No API key configured.[/red]")
        console.print("Set one in ~/.nanobot/config.json under providers section")
        raise typer.Exit(1)

    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(model),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=provider_name,
    )


# ============================================================================
# Gateway / Server —— 网关服务，类比 Go 中的 http.ListenAndServe 启动服务
# 这是 nanobot 的“常驻服务模式”，同时运行：
#   - Agent 循环（处理消息）
#   - 多个渠道（Telegram/WhatsApp/Slack 等）
#   - 定时任务（Cron）
#   - 心跳服务（Heartbeat）
# 类比 Go 的 errgroup.Group 并发启动多个服务
# ============================================================================


@app.command()
def gateway(
    port: int = typer.Option(18790, "--port", "-p", help="Gateway port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """启动 nanobot 网关服务（常驻模式）。

    类比 Go：
        func main() {
            g, ctx := errgroup.WithContext(context.Background())
            g.Go(func() { agent.Run(ctx) })       // Agent 主循环
            g.Go(func() { channels.StartAll(ctx) }) // 所有渠道
            g.Go(func() { cron.Start(ctx) })        // 定时任务
            g.Go(func() { heartbeat.Start(ctx) })   // 心跳
            g.Wait()
        }
    """
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.channels.manager import ChannelManager
    from nanobot.config.loader import get_data_dir, load_config
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    from nanobot.heartbeat.service import HeartbeatService
    from nanobot.session.manager import SessionManager

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    console.print(f"{__logo__} Starting nanobot gateway on port {port}...")

    config = load_config()
    sync_workspace_templates(config.workspace_path)

    # 消息总线 —— 整个系统的“神经中枢”
    # 类比 Go 的双向 channel：
    #   inbound  chan Message  // 用户消息进入
    #   outbound chan Message  // agent 回复发出
    # 所有渠道(Telegram/WhatsApp等)往 bus 发消息，Agent 从 bus 取消息处理
    bus = MessageBus()
    provider = _make_provider(config)  # LLM 提供者（调用 AI 模型）
    session_manager = SessionManager(config.workspace_path)  # 会话管理，保存对话历史

    # 定时任务服务 —— 类比 Go 的 robfig/cron
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # 创建 Agent 主循环 —— 这是“大脑”，接收消息、调用 LLM、执行工具
    # 类比 Go：
    #   agent := NewAgentLoop(AgentConfig{
    #       Bus: bus, Provider: provider, Model: "gpt-4o", ...
    #   })
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        reasoning_effort=config.agents.defaults.reasoning_effort,
        brave_api_key=config.tools.web.search.api_key or None,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
    )

    # 设置定时任务回调 —— 当定时任务触发时，通过 agent 执行
    # 类比 Go 的回调模式：cron.AddFunc("*/5 * * * *", func() { agent.Process(msg) })
    async def on_cron_job(job: CronJob) -> str | None:
        """定时任务触发时的回调函数，用 agent 执行任务指令。"""
        from nanobot.agent.tools.message import MessageTool
        reminder_note = (
            "[Scheduled Task] Timer finished.\n\n"
            f"Task '{job.name}' has been triggered.\n"
            f"Scheduled instruction: {job.payload.message}"
        )

        response = await agent.process_direct(
            reminder_note,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to or "direct",
        )

        # 检查 agent 是否已经通过 message 工具主动发送了消息
        # 如果是，就不重复发送了（避免重复回复）
        message_tool = agent.tools.get("message")
        if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
            return response

        # 如果任务配置了“投递”，将结果通过消息总线发送到指定渠道/用户
        # 类比 Go：outbound <- Message{Channel: "telegram", ChatID: "12345", Content: resp}
        if job.payload.deliver and job.payload.to and response:
            from nanobot.bus.events import OutboundMessage
            await bus.publish_outbound(OutboundMessage(
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to,
                content=response
            ))
        return response
    cron.on_job = on_cron_job  # 注册回调，类比 Go 中 cron.SetCallback(onCronJob)

    # 渠道管理器 —— 统一管理所有通信渠道的启动/停止
    # 类比 Go 中的多个 goroutine 分别监听不同服务：
    #   go telegramBot.Listen(ctx)
    #   go whatsappBridge.Listen(ctx)
    channels = ChannelManager(config, bus)

    def _pick_heartbeat_target() -> tuple[str, str]:
        """为心跳消息选择一个可达的渠道/聊天目标。

        类比 Go：从活跃会话列表中找到第一个可用的外部渠道，
        像从 map[string]Session 中遍历找到第一个 enabled 的 channel。
        找不到就回退到 "cli:direct"。
        """
        enabled = set(channels.enabled_channels)
        # Prefer the most recently updated non-internal session on an enabled channel.
        for item in session_manager.list_sessions():
            key = item.get("key") or ""
            if ":" not in key:
                continue
            channel, chat_id = key.split(":", 1)
            if channel in {"cli", "system"}:
                continue
            if channel in enabled and chat_id:
                return channel, chat_id
        # Fallback keeps prior behavior but remains explicit.
        return "cli", "direct"

    # 创建心跳服务 —— 定期检查系统状态，类比 Go 的 health check ticker
    #   ticker := time.NewTicker(5 * time.Minute)
    #   go func() { for range ticker.C { checkHealth() } }()
    async def on_heartbeat_execute(tasks: str) -> str:
        """心跳触发时：通过 agent 执行检查任务。"""
        channel, chat_id = _pick_heartbeat_target()

        async def _silent(*_args, **_kwargs):
            pass

        return await agent.process_direct(
            tasks,
            session_key="heartbeat",
            channel=channel,
            chat_id=chat_id,
            on_progress=_silent,
        )

    async def on_heartbeat_notify(response: str) -> None:
        """心跳结果投递到用户的渠道（如果有外部渠道可用的话）。"""
        from nanobot.bus.events import OutboundMessage
        channel, chat_id = _pick_heartbeat_target()
        if channel == "cli":
            return  # No external channel available to deliver to
        await bus.publish_outbound(OutboundMessage(channel=channel, chat_id=chat_id, content=response))

    hb_cfg = config.gateway.heartbeat
    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        provider=provider,
        model=agent.model,
        on_execute=on_heartbeat_execute,
        on_notify=on_heartbeat_notify,
        interval_s=hb_cfg.interval_s,
        enabled=hb_cfg.enabled,
    )

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

    console.print(f"[green]✓[/green] Heartbeat: every {hb_cfg.interval_s}s")

    # 启动所有服务 —— asyncio.gather ≈ Go errgroup.Go
    # 同时启动 Agent、所有渠道、Cron、Heartbeat，等待全部完成
    async def run():
        try:
            await cron.start()       # 启动定时任务调度器
            await heartbeat.start()  # 启动心跳
            await asyncio.gather(    # ≈ Go errgroup.Wait()
                agent.run(),         # Agent 主循环：从 bus 取消息 → 调用 LLM → 发回复
                channels.start_all(),# 所有渠道监听：Telegram bot、WhatsApp webhook 等
            )
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        finally:
            # 优雅关闭所有服务，≈ Go 的 defer + context.Cancel()
            await agent.close_mcp()  # 关闭 MCP 工具服务器连接
            heartbeat.stop()
            cron.stop()
            agent.stop()
            await channels.stop_all()

    asyncio.run(run())  # ≈ Go 的 func main() { run() }，启动事件循环


# ============================================================================
# Agent 命令 —— 最核心的交互入口
# 两种模式：
#   1. 单消息模式：nanobot agent -m "你好"  → 发一条消息，收到回复后退出
#   2. 交互模式： nanobot agent           → 进入 REPL 循环，持续对话
# 类比 Go：
#   单消息 = 直接调用函数并等待返回
#   交互模式 = for { input := readline(); response := process(input); print(response) }
# ============================================================================


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render assistant output as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show nanobot runtime logs during chat"),
):
    """与 Agent 直接对话。

    类比 Go：
        func agentCmd(message string, sessionID string) {
            if message != "" {
                resp := agent.ProcessDirect(message)  // 单消息模式
                fmt.Println(resp)
            } else {
                for {  // 交互模式 REPL
                    input := readline()
                    resp := agent.Process(input)
                    fmt.Println(resp)
                }
            }
        }
    """
    from loguru import logger

    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.loader import get_data_dir, load_config
    from nanobot.cron.service import CronService

    # 加载配置 + 创建基础设施
    config = load_config()
    sync_workspace_templates(config.workspace_path)

    bus = MessageBus()           # 消息总线，≈ Go chan，交互模式下用来在主循环和 agent 间传消息
    provider = _make_provider(config)  # LLM 提供者

    # 定时任务服务（CLI 模式下主要供 agent 的 cron 工具使用）
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)

    # 日志控制：类比 Go 中 log.SetOutput(io.Discard) vs log.SetOutput(os.Stderr)
    if logs:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        reasoning_effort=config.agents.defaults.reasoning_effort,
        brave_api_key=config.tools.web.search.api_key or None,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
    )

    # “思考中”动画上下文管理器
    # ≈ Go 中的 spinner := yacspin.New(cfg); spinner.Start(); defer spinner.Stop()
    # 开启日志时不显示 spinner，避免和日志输出混杂
    def _thinking_ctx():
        if logs:
            from contextlib import nullcontext
            return nullcontext()
        # Animated spinner is safe to use with prompt_toolkit input handling
        return console.status("[dim]nanobot is thinking...[/dim]", spinner="dots")

    # 进度回调 —— agent 在处理过程中会通过这个回调报告当前正在做什么
    # 类比 Go：onProgress := func(msg string) { fmt.Printf("  → %s\n", msg) }
    async def _cli_progress(content: str, *, tool_hint: bool = False) -> None:
        ch = agent_loop.channels_config
        if ch and tool_hint and not ch.send_tool_hints:
            return
        if ch and not tool_hint and not ch.send_progress:
            return
        console.print(f"  [dim]↳ {content}[/dim]")

    if message:
       #单消息模式：直接调用，无需总线
        # Single message mode — direct call, no bus needed
        async def run_once():
            with _thinking_ctx():
                response = await agent_loop.process_direct(message, session_id, on_progress=_cli_progress)
            _print_agent_response(response, render_markdown=markdown)
            await agent_loop.close_mcp()

        asyncio.run(run_once())
    else:
        # Interactive mode — route through bus like other channels
        from nanobot.bus.events import InboundMessage
        _init_prompt_session()
        console.print(f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n")

        if ":" in session_id:
            cli_channel, cli_chat_id = session_id.split(":", 1)
        else:
            cli_channel, cli_chat_id = "cli", session_id

        def _exit_on_sigint(signum, frame):
            _restore_terminal()
            console.print("\nGoodbye!")
            os._exit(0)
    #注册中断信号，执行推出函数
        signal.signal(signal.SIGINT, _exit_on_sigint)

        async def run_interactive():    # 定义一个异步函数，协程，使用async.run来运行这个函数，或者awit来执行
            bus_task = asyncio.create_task(agent_loop.run())
            turn_done = asyncio.Event() # 类比 Go 中的 channel，用于协程间的同步
            turn_done.set() # 初始化为已完成状态，避免首次等待阻塞
            turn_response: list[str] = []

            async def _consume_outbound():
                while True:
                    try:
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                        if msg.metadata.get("_progress"):
                            is_tool_hint = msg.metadata.get("_tool_hint", False)        # 检测工具是否命中，是根据msg来判断
                            ch = agent_loop.channels_config
                            if ch and is_tool_hint and not ch.send_tool_hints:
                                pass
                            elif ch and not is_tool_hint and not ch.send_progress:
                                pass
                            else:
                                console.print(f"  [dim]↳ {msg.content}[/dim]")
                        elif not turn_done.is_set():
                            if msg.content:
                                turn_response.append(msg.content)
                            turn_done.set()
                        elif msg.content:
                            console.print()
                            _print_agent_response(msg.content, render_markdown=markdown)
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            outbound_task = asyncio.create_task(_consume_outbound())

            try:
                while True:
                    try:
                        _flush_pending_tty_input()
                        user_input = await _read_interactive_input_async()
                        # 首先判断用户是否输出了命令
                        command = user_input.strip()
                        if not command:
                            continue

                        if _is_exit_command(command):
                            _restore_terminal()
                            console.print("\nGoodbye!")
                            break

                        turn_done.clear()
                        turn_response.clear()

                        await bus.publish_inbound(InboundMessage(
                            channel=cli_channel,
                            sender_id="user",
                            chat_id=cli_chat_id,
                            content=user_input,
                        ))

                        with _thinking_ctx():
                            await turn_done.wait()

                        if turn_response:
                            _print_agent_response(turn_response[0], render_markdown=markdown)
                    except KeyboardInterrupt:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    except EOFError:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
            finally:
                agent_loop.stop()
                outbound_task.cancel()
                await asyncio.gather(bus_task, outbound_task, return_exceptions=True)
                await agent_loop.close_mcp()

        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status():
    """Show channel status."""
    from nanobot.config.loader import load_config

    config = load_config()

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled", style="green")
    table.add_column("Configuration", style="yellow")

    # WhatsApp
    wa = config.channels.whatsapp
    table.add_row(
        "WhatsApp",
        "✓" if wa.enabled else "✗",
        wa.bridge_url
    )

    dc = config.channels.discord
    table.add_row(
        "Discord",
        "✓" if dc.enabled else "✗",
        dc.gateway_url
    )

    # Feishu
    fs = config.channels.feishu
    fs_config = f"app_id: {fs.app_id[:10]}..." if fs.app_id else "[dim]not configured[/dim]"
    table.add_row(
        "Feishu",
        "✓" if fs.enabled else "✗",
        fs_config
    )

    # Mochat
    mc = config.channels.mochat
    mc_base = mc.base_url or "[dim]not configured[/dim]"
    table.add_row(
        "Mochat",
        "✓" if mc.enabled else "✗",
        mc_base
    )

    # Telegram
    tg = config.channels.telegram
    tg_config = f"token: {tg.token[:10]}..." if tg.token else "[dim]not configured[/dim]"
    table.add_row(
        "Telegram",
        "✓" if tg.enabled else "✗",
        tg_config
    )

    # Slack
    slack = config.channels.slack
    slack_config = "socket" if slack.app_token and slack.bot_token else "[dim]not configured[/dim]"
    table.add_row(
        "Slack",
        "✓" if slack.enabled else "✗",
        slack_config
    )

    # DingTalk
    dt = config.channels.dingtalk
    dt_config = f"client_id: {dt.client_id[:10]}..." if dt.client_id else "[dim]not configured[/dim]"
    table.add_row(
        "DingTalk",
        "✓" if dt.enabled else "✗",
        dt_config
    )

    # QQ
    qq = config.channels.qq
    qq_config = f"app_id: {qq.app_id[:10]}..." if qq.app_id else "[dim]not configured[/dim]"
    table.add_row(
        "QQ",
        "✓" if qq.enabled else "✗",
        qq_config
    )

    # Email
    em = config.channels.email
    em_config = em.imap_host if em.imap_host else "[dim]not configured[/dim]"
    table.add_row(
        "Email",
        "✓" if em.enabled else "✗",
        em_config
    )

    console.print(table)


def _get_bridge_dir() -> Path:
    """Get the bridge directory, setting it up if needed."""
    import shutil
    import subprocess

    # User's bridge location
    user_bridge = Path.home() / ".nanobot" / "bridge"

    # Check if already built
    if (user_bridge / "dist" / "index.js").exists():
        return user_bridge

    # Check for npm
    if not shutil.which("npm"):
        console.print("[red]npm not found. Please install Node.js >= 18.[/red]")
        raise typer.Exit(1)

    # Find source bridge: first check package data, then source dir
    pkg_bridge = Path(__file__).parent.parent / "bridge"  # nanobot/bridge (installed)
    src_bridge = Path(__file__).parent.parent.parent / "bridge"  # repo root/bridge (dev)

    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge

    if not source:
        console.print("[red]Bridge source not found.[/red]")
        console.print("Try reinstalling: pip install --force-reinstall nanobot")
        raise typer.Exit(1)

    console.print(f"{__logo__} Setting up bridge...")

    # Copy to user directory
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))

    # Install and build
    try:
        console.print("  Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=user_bridge, check=True, capture_output=True)

        console.print("  Building...")
        subprocess.run(["npm", "run", "build"], cwd=user_bridge, check=True, capture_output=True)

        console.print("[green]✓[/green] Bridge ready\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        if e.stderr:
            console.print(f"[dim]{e.stderr.decode()[:500]}[/dim]")
        raise typer.Exit(1)

    return user_bridge


@channels_app.command("login")
def channels_login():
    """Link device via QR code."""
    import subprocess

    from nanobot.config.loader import load_config

    config = load_config()
    bridge_dir = _get_bridge_dir()

    console.print(f"{__logo__} Starting bridge...")
    console.print("Scan the QR code to connect.\n")

    env = {**os.environ}
    if config.channels.whatsapp.bridge_token:
        env["BRIDGE_TOKEN"] = config.channels.whatsapp.bridge_token

    try:
        subprocess.run(["npm", "start"], cwd=bridge_dir, check=True, env=env)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Bridge failed: {e}[/red]")
    except FileNotFoundError:
        console.print("[red]npm not found. Please install Node.js.[/red]")


# ============================================================================
# Cron Commands
# ============================================================================

cron_app = typer.Typer(help="Manage scheduled tasks")
app.add_typer(cron_app, name="cron")


@cron_app.command("list")
def cron_list(
    all: bool = typer.Option(False, "--all", "-a", help="Include disabled jobs"),
):
    """List scheduled jobs."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)

    jobs = service.list_jobs(include_disabled=all)

    if not jobs:
        console.print("No scheduled jobs.")
        return

    table = Table(title="Scheduled Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Schedule")
    table.add_column("Status")
    table.add_column("Next Run")

    import time
    from datetime import datetime as _dt
    from zoneinfo import ZoneInfo
    for job in jobs:
        # Format schedule
        if job.schedule.kind == "every":
            sched = f"every {(job.schedule.every_ms or 0) // 1000}s"
        elif job.schedule.kind == "cron":
            sched = f"{job.schedule.expr or ''} ({job.schedule.tz})" if job.schedule.tz else (job.schedule.expr or "")
        else:
            sched = "one-time"

        # Format next run
        next_run = ""
        if job.state.next_run_at_ms:
            ts = job.state.next_run_at_ms / 1000
            try:
                tz = ZoneInfo(job.schedule.tz) if job.schedule.tz else None
                next_run = _dt.fromtimestamp(ts, tz).strftime("%Y-%m-%d %H:%M")
            except Exception:
                next_run = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))

        status = "[green]enabled[/green]" if job.enabled else "[dim]disabled[/dim]"

        table.add_row(job.id, job.name, sched, status, next_run)

    console.print(table)


@cron_app.command("add")
def cron_add(
    name: str = typer.Option(..., "--name", "-n", help="Job name"),
    message: str = typer.Option(..., "--message", "-m", help="Message for agent"),
    every: int = typer.Option(None, "--every", "-e", help="Run every N seconds"),
    cron_expr: str = typer.Option(None, "--cron", "-c", help="Cron expression (e.g. '0 9 * * *')"),
    tz: str | None = typer.Option(None, "--tz", help="IANA timezone for cron (e.g. 'America/Vancouver')"),
    at: str = typer.Option(None, "--at", help="Run once at time (ISO format)"),
    deliver: bool = typer.Option(False, "--deliver", "-d", help="Deliver response to channel"),
    to: str = typer.Option(None, "--to", help="Recipient for delivery"),
    channel: str = typer.Option(None, "--channel", help="Channel for delivery (e.g. 'telegram', 'whatsapp')"),
):
    """Add a scheduled job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronSchedule

    if tz and not cron_expr:
        console.print("[red]Error: --tz can only be used with --cron[/red]")
        raise typer.Exit(1)

    # Determine schedule type
    if every:
        schedule = CronSchedule(kind="every", every_ms=every * 1000)
    elif cron_expr:
        schedule = CronSchedule(kind="cron", expr=cron_expr, tz=tz)
    elif at:
        import datetime
        dt = datetime.datetime.fromisoformat(at)
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
    else:
        console.print("[red]Error: Must specify --every, --cron, or --at[/red]")
        raise typer.Exit(1)

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)

    try:
        job = service.add_job(
            name=name,
            schedule=schedule,
            message=message,
            deliver=deliver,
            to=to,
            channel=channel,
        )
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e

    console.print(f"[green]✓[/green] Added job '{job.name}' ({job.id})")


@cron_app.command("remove")
def cron_remove(
    job_id: str = typer.Argument(..., help="Job ID to remove"),
):
    """Remove a scheduled job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)

    if service.remove_job(job_id):
        console.print(f"[green]✓[/green] Removed job {job_id}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("enable")
def cron_enable(
    job_id: str = typer.Argument(..., help="Job ID"),
    disable: bool = typer.Option(False, "--disable", help="Disable instead of enable"),
):
    """Enable or disable a job."""
    from nanobot.config.loader import get_data_dir
    from nanobot.cron.service import CronService

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)

    job = service.enable_job(job_id, enabled=not disable)
    if job:
        status = "disabled" if disable else "enabled"
        console.print(f"[green]✓[/green] Job '{job.name}' {status}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("run")
def cron_run(
    job_id: str = typer.Argument(..., help="Job ID to run"),
    force: bool = typer.Option(False, "--force", "-f", help="Run even if disabled"),
):
    """Manually run a job."""
    from loguru import logger

    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.loader import get_data_dir, load_config
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    logger.disable("nanobot")

    config = load_config()
    provider = _make_provider(config)
    bus = MessageBus()
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        temperature=config.agents.defaults.temperature,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        reasoning_effort=config.agents.defaults.reasoning_effort,
        brave_api_key=config.tools.web.search.api_key or None,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
    )

    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)

    result_holder = []

    async def on_job(job: CronJob) -> str | None:
        response = await agent_loop.process_direct(
            job.payload.message,
            session_key=f"cron:{job.id}",
            channel=job.payload.channel or "cli",
            chat_id=job.payload.to or "direct",
        )
        result_holder.append(response)
        return response

    service.on_job = on_job

    async def run():
        return await service.run_job(job_id, force=force)

    if asyncio.run(run()):
        console.print("[green]✓[/green] Job executed")
        if result_holder:
            _print_agent_response(result_holder[0], render_markdown=True)
    else:
        console.print(f"[red]Failed to run job {job_id}[/red]")


# ============================================================================
# Status Commands
# ============================================================================


@app.command()
def status():
    """Show nanobot status."""
    from nanobot.config.loader import get_config_path, load_config

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} nanobot Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        from nanobot.providers.registry import PROVIDERS

        console.print(f"Model: {config.agents.defaults.model}")

        # Check API keys from registry
        for spec in PROVIDERS:
            p = getattr(config.providers, spec.name, None)
            if p is None:
                continue
            if spec.is_oauth:
                console.print(f"{spec.label}: [green]✓ (OAuth)[/green]")
            elif spec.is_local:
                # Local deployments show api_base instead of api_key
                if p.api_base:
                    console.print(f"{spec.label}: [green]✓ {p.api_base}[/green]")
                else:
                    console.print(f"{spec.label}: [dim]not set[/dim]")
            else:
                has_key = bool(p.api_key)
                console.print(f"{spec.label}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")


# ============================================================================
# OAuth Login
# ============================================================================

provider_app = typer.Typer(help="Manage providers")
app.add_typer(provider_app, name="provider")


_LOGIN_HANDLERS: dict[str, callable] = {}


def _register_login(name: str):
    def decorator(fn):
        _LOGIN_HANDLERS[name] = fn
        return fn
    return decorator


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="OAuth provider (e.g. 'openai-codex', 'github-copilot')"),
):
    """Authenticate with an OAuth provider."""
    from nanobot.providers.registry import PROVIDERS

    key = provider.replace("-", "_")
    spec = next((s for s in PROVIDERS if s.name == key and s.is_oauth), None)
    if not spec:
        names = ", ".join(s.name.replace("_", "-") for s in PROVIDERS if s.is_oauth)
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]  Supported: {names}")
        raise typer.Exit(1)

    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Login not implemented for {spec.label}[/red]")
        raise typer.Exit(1)

    console.print(f"{__logo__} OAuth Login - {spec.label}\n")
    handler()


@_register_login("openai_codex")
def _login_openai_codex() -> None:
    try:
        from oauth_cli_kit import get_token, login_oauth_interactive
        token = None
        try:
            token = get_token()
        except Exception:
            pass
        if not (token and token.access):
            console.print("[cyan]Starting interactive OAuth login...[/cyan]\n")
            token = login_oauth_interactive(
                print_fn=lambda s: console.print(s),
                prompt_fn=lambda s: typer.prompt(s),
            )
        if not (token and token.access):
            console.print("[red]✗ Authentication failed[/red]")
            raise typer.Exit(1)
        console.print(f"[green]✓ Authenticated with OpenAI Codex[/green]  [dim]{token.account_id}[/dim]")
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)


@_register_login("github_copilot")
def _login_github_copilot() -> None:
    import asyncio

    console.print("[cyan]Starting GitHub Copilot device flow...[/cyan]\n")

    async def _trigger():
        from litellm import acompletion
        await acompletion(model="github_copilot/gpt-4o", messages=[{"role": "user", "content": "hi"}], max_tokens=1)

    try:
        asyncio.run(_trigger())
        console.print("[green]✓ Authenticated with GitHub Copilot[/green]")
    except Exception as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
