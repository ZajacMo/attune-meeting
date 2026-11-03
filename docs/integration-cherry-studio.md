# Attune in Cherry Studio

Cherry Studio speaks MCP natively, so Attune is a first-class citizen there —
no plugin, no code changes.

## 1. Start the server

```bash
pip install "attune-meeting[server]"
attune-mcp --workspace ~/attune-data
```

That speaks MCP over stdio. For a shared or remote setup, run it over HTTP
instead:

```bash
attune-mcp --http --host 127.0.0.1 --port 8788
```

## 2. Register it in Cherry Studio

Open **Settings → MCP Servers → Add server**, and use a local command:

```json
{
  "mcpServers": {
    "attune": {
      "command": "attune-mcp",
      "args": ["--workspace", "/absolute/path/to/attune-data"],
      "env": {"VEXA_API_KEY": "vxa_..."}
    }
  }
}
```

On Windows the command may need the full path to the shim:

```json
{
  "mcpServers": {
    "attune": {
      "command": "C:\\path\\to\\python\\Scripts\\attune-mcp.exe",
      "args": ["--workspace", "C:\\Users\\you\\attune-data"]
    }
  }
}
```

For the HTTP transport, use the URL form instead:

```json
{
  "mcpServers": {
    "attune": {
      "url": "http://127.0.0.1:8788/mcp"
    }
  }
}
```

## 3. Use it

Cherry Studio's model can call the tools directly. A prompt that works:

> 用 attune 整理 `D:\meetings\weekly.wav`，项目叫 `my-project`。
> 如果有说话人需要确认，列出候选身份让我选。

If Cherry Studio cannot read the file itself, drop it somewhere it can reach
and pass the absolute path to `transcribe_audio`.

## Notes

- The ASR extra is optional but usually wanted: `pip install "attune-meeting[asr-funasr]"`.
  Without it, `transcribe_audio` returns a clear error saying no engine is
  available.
- The **project id** is the identity scope. Use one id per ongoing set of
  meetings so people are recognised across sessions — a different id means a
  different set of people.
- Attune does not need a model key. Minutes use the deterministic generator by
  default, which keeps the integration free and offline.
