## qwen3-online-mcp-speech-agent

## Qwen3-235B-A22b+MCP语音助手

### 进入client目录修改env环境变量，运行 python voice-qwen3-235b.py
【Qwen3 235B 语音助手 接入MCP】 https://www.bilibili.com/video/BV12YGhz5EGf/?share_source=copy_web&vd_source=245c190fe77b507d57968a57b3d6f9cf


### 支持了SSE MCP的配置，增加了isActive节点，默认配置了一个高德地图的MCP托管，请到百炼生成key替换下面的xxx。（该高德地图可以用来查询天气！）
{
  "mcpServers": {
    "music_player": {
      "isActive":true,
      "command": "python",
      "args": [
        "../server/mplayer/main.py"
      ]
    },
    "amap-maps": {
      "isActive":true,
      "type": "sse",
      "url": "https://mcp.api-inference.modelscope.cn/sse/xxx"
      }
  }
}
