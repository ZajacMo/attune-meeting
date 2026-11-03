# Attune

**一个知道谁在说话的会议助手——并且下次还记得他。**

Attune 把一段会议录音变成两份文档：

1. **会议纪要**——按人整理每个人的发言大纲，外加会议里产生的具体待办事项。
2. **全程转录**——每一句话都有时间戳，并归属到真实姓名。

难点从来不是转写，而是**身份**。说话人分离引擎只能说出"说话人 0""说话人 1"——
这些标签离开那一段录音就没有意义。Attune 维护一份**按项目划分的声纹名册**，
所以同一个人在下周那场会议里还会被认出来；而且当匹配不够确定时它**绝不
静默猜测**，而是把选择权交回给人。

---

## 它补的是哪一块

所有主流分离库都有同一个边界：`pyannote.audio` 输出 `speaker_0`/`speaker_1`；
FunASR 输出局部 `spk` 索引——它自己的文档写着*"Indices are local to a
recording, not known-person identities."* 你拿到的是话轮结构，不是人。

Attune 补上另一半：

| 阶段 | Attune 做什么 |
| --- | --- |
| **分离** | 把音频切成话轮并按声纹聚类（默认 FunASR + CAM++，可选 WhisperX）。 |
| **匹配** | 拿每个声纹和项目名册比对。 |
| **决策** | 匹配确信 → 静默沿用该身份；接近但不确定 → **问用户**：合并到已有身份，还是新建一个； |
| **记住** | 把声纹登记到项目名下，下次会议自动认出同一个人。 |
| **产出** | 用真实姓名生成纪要和转录。 |

名册挂在**项目**上而不是会话上——这正是跨会话识别能成立的原因。

---

## 安装

```bash
pip install attune-meeting         # 仅核心库
pip install "attune-meeting[server]"       # + MCP 服务与 HTTP API
pip install "attune-meeting[asr-funasr]"   # + 中文 ASR 与说话人聚类
```

ASR 扩展是刻意做成可选的：核心库、身份名册和文档生成足够轻，可以塞进任何东西。

### 默认走中文优先路线

FunASR 一次调用就给出 ASR、VAD、标点和说话人聚类：

```python
from attune.core.engines import FunASREngine

engine = FunASREngine(model="paraformer-zh", device="cpu")   # fsmn-vad + ct-punc + cam++
```

---

## 快速开始

```bash
# 转写、确认说话人、打印纪要
attune transcribe meeting.wav --project my-project --title "产品周会"

# 看这个项目认得哪些人
attune speakers my-project

# 回答"合并还是新建"
attune resolve my-project ses_abc123 \
  --engine-speaker spk_0 --action rename --display-name "张三"

# 打印两份文档
attune minutes   my-project ses_abc123
attune transcript my-project ses_abc123
```

Python 用法：

```python
from attune.core.engines import FunASREngine
from attune.core.identity import SpeakerRegistry
from attune.core.pipeline import MeetingPipeline, SpeakerDecision
from attune.core.store import Workspace

workspace = Workspace("data")
registry = workspace.load_registry("my-project")
pipeline = MeetingPipeline(FunASREngine(), registry)

result = pipeline.process("meeting.wav", session_id="ses_1", title="产品周会")

# 还需要用户拍板的部分
for item in result.resolutions:
    if not item.resolved:
        print(item.engine_speaker, item.kind, item.suggested_name,
              [(c.display_name, round(c.score, 2)) for c in item.candidates])

pipeline.resolve(result, [
    SpeakerDecision(engine_speaker="spk_0", action="rename", display_name="张三"),
])

workspace.save_session(result)
workspace.save_registry(registry)
print(result.minutes_markdown)
```

---

## 身份模型

```json
{
  "version": 1,
  "project_id": "my-project",
  "high_threshold": 0.68,
  "low_threshold": 0.53,
  "profiles": [
    {
      "identity_id": "spk_1a2b3c4d",
      "display_name": "张三",
      "aliases": ["张经理"],
      "embeddings": [[0.12, -0.44, "..."]],
      "appearances": 7,
      "total_seconds": 512.4,
      "confirmed": true
    }
  ],
  "bindings": [
    {"session_id": "ses_1", "engine_speaker": "spk_0", "identity_id": "spk_1a2b3c4d"}
  ]
}
```

匹配有三种结果：

- **known**——相似度 ≥ `high_threshold` → 静默沿用该身份。
- **ambiguous**——落在两个阈值之间 → 交给用户决定合并还是新建。Attune 在这里不猜。
- **new**——相似度 < `low_threshold` → 新建身份并等待命名。

> **阈值是起点，不是定律。** 出厂默认值（0.68 / 0.53）是在真实中文会议音频上实测出来的——
> 数据、方法与可复现脚本见 [`docs/voiceprint-calibration.md`](docs/voiceprint-calibration.md)。
> 换了麦克风和房间，声纹本身就会变，请在自有音频上重新实测再决定是否沿用。

---

## 接口

同一套操作提供两种形态，宿主用哪个都行。

### MCP 服务

```bash
attune-mcp --workspace ./data          # stdio
attune-mcp --http --port 8788           # streamable HTTP
```

工具：`create_session`、`transcribe_audio`、`get_transcript`、`get_minutes`、
`list_speakers`、`get_pending_speaker_decisions`、`resolve_speaker`。

### HTTP API

```bash
attune serve --host 127.0.0.1 --port 8787
```

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/v1/projects/{project}/sessions` | 创建会话 |
| `POST` | `/v1/projects/{project}/sessions/{session}/audio` | 上传音频（裸 body）并处理 |
| `GET` | `/v1/projects/{project}/sessions/{session}/transcript` | 转录（`?format=markdown` 出 Markdown） |
| `GET` | `/v1/projects/{project}/sessions/{session}/minutes` | 纪要（`?format=markdown` 出 Markdown） |
| `GET` | `/v1/projects/{project}/sessions/{session}/speakers` | 绑定关系与待确认项 |
| `POST` | `/v1/projects/{project}/sessions/{session}/speakers/resolve` | 回答合并/新建 |
| `GET` | `/v1/projects/{project}/speakers` | 项目认识的所有人 |

完整契约见 [docs/protocol.md](docs/protocol.md)。

---

## 加入线上会议

Attune 自己不写会议 bot，它用 [Vexa](https://github.com/Vexa-ai/vexa)（Apache-2.0）
拿音频，其余全部自己做。

```bash
export VEXA_API_KEY=vxa_...

attune vexa send --url https://meet.google.com/abc-defg-hij --language zh
attune vexa transcript --url https://meet.google.com/abc-defg-hij
attune vexa stop --url https://meet.google.com/abc-defg-hij
```

Google Meet、Microsoft Teams、Zoom、Jitsi 的链接都能认。Vexa 自带的转写默认
**关闭**——Attune 自己转写和分离，同时开等于白付一次钱。想要对照结果时加
`--transcribe`。

---

## 框架接入指南

- [Cherry Studio](docs/integration-cherry-studio.md)——加成 MCP 服务
- [Hermes Agent](docs/integration-hermes.md)——Python 包与 HTTP API
- [OpenClaw](docs/integration-openclaw.md)——skill 包形态
- [CyberColleague](docs/integration-cybercolleague.md)——桌面工作台

---

## 开发

```bash
uv sync --extra dev --extra server
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

## 许可

Apache-2.0，见 [LICENSE](LICENSE)。

可选 ASR 扩展用到的模型权重有各自的许可——要打包进产品请先看模型卡。
