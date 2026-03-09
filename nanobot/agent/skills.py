"""Skills loader for agent capabilities.

该模块负责在工作区和内置目录中查找 SKILL.md 文件，
解析其 frontmatter 元数据，并根据依赖（可执行文件、环境变量）判断技能是否可用。
代码中使用 pathlib.Path 进行路径操作，使用 shutil.which 检查可执行程序是否存在。
所有函数均保留原有行为，仅添加逐行中文注释以帮助来自 Go 背景的开发者理解。
"""

import json  # 处理 JSON 字符串/对象
import os  # 访问环境变量等操作系统功能
import re  # 正则，用于解析 frontmatter
import shutil  # 提供 which() 等工具，用于检查可执行文件是否可用
from pathlib import Path  # 面向对象的文件路径操作

# 默认内置技能目录，基于本文件的父目录的 parent/skills
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"


class SkillsLoader:
    """Loader for agent skills.

    Skills are markdown files (SKILL.md) located under either the workspace's
    `skills/` directory or the built-in skills directory. 每个技能目录包含
    一个 SKILL.md，可能在文件头部包含 frontmatter（YAML 或 JSON）用于
    声明依赖（如需要的命令行工具、环境变量）和描述信息。
    """

    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None):
        # 将工作区路径保存为 Path 对象
        self.workspace = workspace
        # workspace 下的 skills 目录路径
        self.workspace_skills = workspace / "skills"
        # 内置技能目录，若外部传入则使用传入值
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """列出所有技能。

        如果 filter_unavailable 为 True，则会用 _check_requirements 过滤掉
        因未满足外部依赖而不可用的技能。
        返回值为字典列表，每个字典包含 name/path/source。
        """
        skills = []

        # 优先检查工作区的技能目录（本地覆盖内置）
        if self.workspace_skills.exists():
            for skill_dir in self.workspace_skills.iterdir():
                # 仅处理目录项
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    # 如果找到 SKILL.md，则记录其路径和来源
                    if skill_file.exists():
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "workspace"})

        # 检查内置技能目录，避免重复添加已在 workspace 中存在的技能
        if self.builtin_skills and self.builtin_skills.exists():
            for skill_dir in self.builtin_skills.iterdir():
                if skill_dir.is_dir():
                    skill_file = skill_dir / "SKILL.md"
                    # 仅当 workspace 中没有同名技能时才添加内置技能
                    if skill_file.exists() and not any(s["name"] == skill_dir.name for s in skills):
                        skills.append({"name": skill_dir.name, "path": str(skill_file), "source": "builtin"})

        # 根据需求过滤技能（例如缺少 CLI 或 ENV 时视为不可用）
        if filter_unavailable:
            return [s for s in skills if self._check_requirements(self._get_skill_meta(s["name"]))]
        return skills

    def load_skill(self, name: str) -> str | None:
        """按名称加载技能内容，优先从工作区，然后从内置目录读取 SKILL.md。

        返回 SKILL.md 的文本内容，若不存在返回 None。
        """
        # 先在 workspace 中查找
        workspace_skill = self.workspace_skills / name / "SKILL.md"
        if workspace_skill.exists():
            return workspace_skill.read_text(encoding="utf-8")

        # 再查找内置目录
        if self.builtin_skills:
            builtin_skill = self.builtin_skills / name / "SKILL.md"
            if builtin_skill.exists():
                return builtin_skill.read_text(encoding="utf-8")

        # 未找到则返回 None
        return None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """加载指定技能的内容，用于注入到 agent 的上下文中。

        对每个技能，去掉 frontmatter（如果有），并以"### Skill: name"为标题拼接。
        返回拼接后的字符串，若没有任何技能则返回空字符串。
        """
        parts = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                # 去掉 YAML/JSON frontmatter，只保留主体内容
                content = self._strip_frontmatter(content)
                parts.append(f"### Skill: {name}\n\n{content}")

        return "\n\n---\n\n".join(parts) if parts else ""

    def build_skills_summary(self) -> str:
        """构建技能汇总（XML 格式），包括名称、描述、路径和是否可用。

        该汇总用于在 system prompt 中告知模型可用的技能和缺失依赖，
        模型在需要时可以使用 read_file 工具查看某个技能的完整内容。
        """
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""

        def escape_xml(s: str) -> str:
            # 将特殊字符替换为 XML 实体，防止注入或格式破坏
            return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for s in all_skills:
            # 对名称和描述进行 XML 转义
            name = escape_xml(s["name"])
            path = s["path"]
            desc = escape_xml(self._get_skill_description(s["name"]))
            # 从 frontmatter 获取技能元数据（如 requires）
            skill_meta = self._get_skill_meta(s["name"]) 
            # 判断技能是否满足依赖
            available = self._check_requirements(skill_meta)

            lines.append(f"  <skill available=\"{str(available).lower()}\">")
            lines.append(f"    <name>{name}</name>")
            lines.append(f"    <description>{desc}</description>")
            lines.append(f"    <location>{path}</location>")

            # 如果不可用，列出缺失的依赖（二进制或环境变量）
            if not available:
                missing = self._get_missing_requirements(skill_meta)
                if missing:
                    lines.append(f"    <requires>{escape_xml(missing)}</requires>")

            lines.append("  </skill>")
        lines.append("</skills>")

        return "\n".join(lines)

    def _get_missing_requirements(self, skill_meta: dict) -> str:
        """返回技能缺失的依赖描述字符串（例如 CLI: git, ENV: OPENAI_API_KEY）。"""
        missing = []
        requires = skill_meta.get("requires", {})
        # 检查需要的可执行程序
        for b in requires.get("bins", []):
            if not shutil.which(b):
                missing.append(f"CLI: {b}")
        # 检查需要的环境变量
        for env in requires.get("env", []):
            if not os.environ.get(env):
                missing.append(f"ENV: {env}")
        return ", ".join(missing)

    def _get_skill_description(self, name: str) -> str:
        """从技能 frontmatter 中提取 description 字段，若无则用技能名作为回退值。"""
        meta = self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return meta["description"]
        return name  # Fallback to skill name

    def _strip_frontmatter(self, content: str) -> str:
        """从 markdown 内容中删除 YAML 前置块（以 --- 包围的部分）。"""
        if content.startswith("---"):
            # 使用非贪婪匹配查找前置块边界
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                # 返回前置块之后的主体内容，并去掉首尾空白
                return content[match.end():].strip()
        return content

    def _parse_nanobot_metadata(self, raw: str) -> dict:
        """解析 frontmatter 中可能包含的 JSON 元数据。

        一些技能在 frontmatter 中以 JSON 形式提供额外键（如 nanobot 或 openclaw），
        该函数尝试解析并返回 nanobot 或 openclaw 键下的对象（或空字典）。
        """
        try:
            data = json.loads(raw)
            # 优先返回 nanobot 字段，回退到 openclaw，若都不存在返回空字典
            return data.get("nanobot", data.get("openclaw", {})) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            # 解析失败时返回空字典
            return {}

    def _check_requirements(self, skill_meta: dict) -> bool:
        """检查技能元数据中声明的依赖是否满足（CLI 可执行和环境变量）。

        返回 True 表示可用，False 表示缺少某些依赖。
        """
        requires = skill_meta.get("requires", {})
        for b in requires.get("bins", []):
            if not shutil.which(b):
                return False
        for env in requires.get("env", []):
            if not os.environ.get(env):
                return False
        return True

    def _get_skill_meta(self, name: str) -> dict:
        """从技能文件的 frontmatter 中获取 nanobot 风格的元数据（如果存在）。"""
        meta = self.get_skill_metadata(name) or {}
        return self._parse_nanobot_metadata(meta.get("metadata", ""))

    def get_always_skills(self) -> list[str]:
        """返回被标记为 always=true 且满足依赖的技能名称列表。"""
        result = []
        for s in self.list_skills(filter_unavailable=True):
            meta = self.get_skill_metadata(s["name"]) or {}
            skill_meta = self._parse_nanobot_metadata(meta.get("metadata", ""))
            # 支持两种位置声明 always（nanobot JSON 或简单 YAML 字段）
            if skill_meta.get("always") or meta.get("always"):
                result.append(s["name"])
        return result

    def get_skill_metadata(self, name: str) -> dict | None:
        """解析并返回技能文件头部的简单元数据字典（非严格 YAML，只做行级分割）。

        返回值为字典（键->字符串值）或 None（未找到文件）。
        注意：此处使用了简单的解析逻辑并非完整 YAML 解析器，这在大多数 SKILL.md 使用简单键值对时足够。
        """
        content = self.load_skill(name)
        if not content:
            return None

        if content.startswith("---"):
            # 匹配开头的前置块并提取其内部文本
            match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if match:
                # 简单地以行分割并按第一个 ':' 分割键值对，这是对 YAML 的简化处理
                metadata = {}
                for line in match.group(1).split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip('"\'')
                return metadata

        return None
