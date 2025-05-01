import asyncio
from contextlib import AsyncExitStack
import json
from dotenv import load_dotenv 
import os

from typing import Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import AsyncOpenAI


import azure.cognitiveservices.speech as speechsdk  

import time  

import json



 
load_dotenv("1.env")  
#client = OpenAI(api_key=os.environ["api_key"], base_url=os.environ["base_url"])
#model="gpt-4o-mini"
#client = OpenAI(api_key="1", base_url="http://localhost:11434/v1") 
Azure_speech_key = os.environ["Azure_speech_key"]  
Azure_speech_region = os.environ["Azure_speech_region"]  
Azure_speech_speaker = os.environ["Azure_speech_speaker"]  
WakeupWord = os.environ["WakeupWord"]  
WakeupModelFile = os.environ["WakeupModelFile"]  

messages = []  

# Set up Azure Speech-to-Text and Text-to-Speech credentials  
speech_key = Azure_speech_key  
service_region = Azure_speech_region  
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)  
# Set up Azure Text-to-Speech language  
speech_config.speech_synthesis_language = "zh-CN"  
# Set up Azure Speech-to-Text language recognition  
speech_config.speech_recognition_language = "zh-CN"  
lang = "zh-CN"  
# Set up the voice configuration  
speech_config.speech_synthesis_voice_name = Azure_speech_speaker  
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)  
connection = speechsdk.Connection.from_speech_synthesizer(speech_synthesizer)  
connection.open(True)  
# Creates an instance of a keyword recognition model. Update this to  
# point to the location of your keyword recognition model.  
model = speechsdk.KeywordRecognitionModel(WakeupModelFile)  
# The phrase your keyword recognition model triggers on.  
keyword = WakeupWord  
# Set up the audio configuration  
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)  
# Create a speech recognizer and start the recognition  
#speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)  
auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["ja-JP", "zh-CN"])  
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config,  
                                               auto_detect_source_language_config=auto_detect_source_language_config)  
unknownCount = 0  
sysmesg = {"role": "system", "content": os.environ["sysprompt_zh-CN"]}  
tts_sentence_end = [ ".", "!", "?", ";", "。", "！", "？", "；", "\n" ]


isListenning=False


def display_text(s):
    print(s)
def speech_to_text():  
    global unknownCount  
    global lang,isListenning  
    print("Please say...")  
    result = speech_recognizer.recognize_once_async().get()  
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:  
        unknownCount = 0  
        isListenning=False
        return result.text  
    elif result.reason == speechsdk.ResultReason.NoMatch:  
        isListenning=False
        unknownCount += 1  
        error = os.environ["sorry_" + lang]  
        text_to_speech(error)  
        return '...'  
    elif result.reason == speechsdk.ResultReason.Canceled:  
        isListenning=False
        return "speech recognizer canceled." 
    

def getVoiceSpeed():  
    return 17  
  
def text_to_speech(text, _lang=None):  
    global lang  
    try:  
        result = buildSpeech(text).get()  
        #result = speech_synthesizer.speak_ssml_async(ssml_text).get()  
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:  
            print("Text-to-speech conversion successful.")  
            return "Done."  
        else:  
            print(f"Error synthesizing audio: {result}")  
            return "Failed."  
    except Exception as ex:  
        print(f"Error synthesizing audio: {ex}")  
        return "Error occured!"  
        
def buildSpeech(text, _lang=None):
    voice_lang = lang  
    voice_name = "zh-CN-XiaoxiaoMultilingualNeural"  
    ssml_text = f'''  
        <speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xmlns:emo="http://www.w3.org/2009/10/emotionml" version="1.0" xml:lang="{lang}"><voice name="{voice_name}"><lang xml:lang="{voice_lang}"><prosody rate="{getVoiceSpeed()}%">{text.replace('*', ' ').replace('#', ' ')}</prosody></lang></voice></speak>  
    '''  
    print(f"{voice_name} {voice_lang}!")  
    return speech_synthesizer.speak_ssml_async(ssml_text) 

modelName=os.environ["model"]
class MCPClient:
    def __init__(self):
        self.playing=False
        self.session: Optional[ClientSession] = None
        self.sessions={}
        self.exit_stack = AsyncExitStack()
        self.tools=[]
        self.messages=[]
        self.client = AsyncOpenAI(
            api_key=os.environ["key"], 
            base_url=os.environ["base_url"]
        )

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    async def connect_to_server(self):
        with open("mcp_server_config.json", "r") as f:
            config = json.load(f)
            print(config["mcpServers"])  
        conf=config["mcpServers"]
        print(conf.keys())
        for key in conf.keys():
            v = conf[key]
            command = v['command']
            args=v['args']
            print(command)
            print(args)
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=None
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio1, write1 = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio1, write1))
            
            await session.initialize()
            
            # 列出可用工具
            response = await session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])
            for tool in tools:
                self.sessions[tool.name]=session
            self.tools=self.tools+tools
            print(self.sessions)

    async def process_query(self, query: str) -> str:
        """使用 LLM 和 MCP 服务器提供的工具处理查询"""
        self.messages.append({"role": "user", "content": query}) 
        messages= self.messages[-10:]
        collected_messages = []
        last_tts_request = None
    
        split=True
    
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in self.tools]
        
        extra_body = {
            # enable thinking, set to False to disable
            "enable_thinking": False,
            # use thinking_budget to contorl num of tokens used for thinking
            # "thinking_budget": 4096
        }
        # 初始化 LLM API 调用
        response_gen = await self.client.chat.completions.create(
            model=modelName,
            messages=[sysmesg]+ messages,
            tools=available_tools,
            stream=True,
            extra_body=extra_body
        )
        final_text = []
        while True:
            result=''
            function_list=[]
            index=0
            async for chunk in response_gen:
                if chunk:
                    
                    delta = chunk.choices[0].delta
                    chunk_message =  delta.content  # 抽取流式message里的内容
              
                    if chunk_message is not None and chunk_message!='':
                        collected_messages.append(chunk_message)
                        result=result+chunk_message
                        final_text.append(chunk_message or "")
                        if chunk_message in tts_sentence_end and split: # 发现分段标记：句子结束
                            text = ''.join(collected_messages).strip() # 构建句子
                            if len(text)>500 or "</think>" in text: #如果这句足够长，后面不再分段朗读了
                                split=False
                            
                            
                            if text != '': # 如果句子里只有 \n or space则跳过朗读
                                print(f"Speech synthesized to speaker for: {text}")
                                #text=text.replace("<think>","让我先来思考一下：")
                                #text=text.replace("</think>","嗯，我想好了，下面是我的回答。")
                                last_tts_request = buildSpeech(text)
                                #last_tts_request = speech_synthesizer.speak_text_async(text)
                                collected_messages.clear()
                                
                    if delta.tool_calls: 
                        for tool_call in delta.tool_calls:
                            if len(function_list) < tool_call.index + 1:  
                                function_list.append({'name': '', 'args': '', 'id': tool_call.id})  
                            if tool_call and tool_call.function.name:  
                                function_list[tool_call.index]['name'] += tool_call.function.name  
                            if tool_call and tool_call.function.arguments:  
                                function_list[tool_call.index]['args'] += tool_call.function.arguments  
                                 
                     
            print(function_list)
            
            if len(function_list)>0:
                findex=0
                tool_calls=[]
                temp_messages=[]
                for func in function_list:
                    function_name = func["name"]
                    print(function_name)
                    function_args = func["args"]
                    toolid=func["id"]
                    if function_name !='':
                        
                        tool_name = function_name
                        tool_args =  json.loads(function_args)
                        
                        # 执行工具调用
                        function_response = await self.sessions[tool_name].call_tool(tool_name, tool_args)
                        #final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                        
                        print(f"MCP: [Calling tool {tool_name} with args {tool_args}]")
                        print(f'⏳result: {function_response}')
                        
                        tool_calls.append({"id":toolid,"function":{"arguments":func["args"], "name":function_name}, "type":"function","index":findex})
                         
                        temp_messages.append(
                            {
                            "tool_call_id": toolid,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response.content[0].text,
                            }
                        )
                        
                        findex+=1
                        
                messages.append({
                            "role":"assistant",
                            "content":'',
                            "tool_calls":tool_calls,
                        })
                
                for m in temp_messages:
                    messages.append(m)
                      
                response_gen =await self.client.chat.completions.create(
                    model=os.environ["model"],
                    messages=messages,
                    tools=available_tools,
                    stream=True,
                    extra_body=extra_body
                )  
            else:
                if result!='':        
                    messages.append({"role": "assistant", "content": result})
                
                if len(collected_messages)>0:
                    text = ''.join(collected_messages).strip() 
                    if text != '': 
                        print(f"Speech synthesized to speaker for: {text}")
                        last_tts_request = buildSpeech(text)
                        collected_messages.clear()
            
                
                if last_tts_request:
                    last_tts_request.get()   
                    
                break
                
        return result
        
    
    async def getPlayerStatus(self):
        result = await self.sessions["isPlaying"].call_tool("isPlaying", {})
        print(f"getPlayerStatus:{result.content[0].text}")
        if result.content[0].text=="true":
            return "playing"
        else:
            return ""
    async def pauseplay(self):
        await self.sessions["pauseplay"].call_tool("pauseplay", {})

    def recognized_cb(self,evt):  
        result = evt.result  
        if result.reason == speechsdk.ResultReason.RecognizedKeyword:  
            print("RECOGNIZED KEYWORD: {}".format(result.text))  
        global done  
        done = True  

    def canceled_cb(self,evt):  
        result = evt.result  
        if result.reason == speechsdk.ResultReason.Canceled:  
            print('CANCELED: {}'.format(result.cancellation_details.reason))  
        global done  
        done = True  
               
    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        global unknownCount ,isListenning 
        while True:  
            keyword_recognizer = speechsdk.KeywordRecognizer()  
            keyword_recognizer.recognized.connect(self.recognized_cb)  
            keyword_recognizer.canceled.connect(self.canceled_cb) 
            first=os.environ["welcome_" + lang]
            display_text(first) 
            if await self.getPlayerStatus()!='playing':
                text_to_speech(first)  
            isListenning=True
            result_future = keyword_recognizer.recognize_once_async(model)  
            while True:  
                result = result_future.get()
                # Read result audio (incl. the keyword).
                if result.reason == speechsdk.ResultReason.RecognizedKeyword:
                    print("Keyword recognized")
                    isListenning=False
                    if await self.getPlayerStatus()=='playing':
                        await self.pauseplay() #被唤醒后，如果有音乐播放则暂停播放
                    break
                time.sleep(0.1)  
                
            display_text("很高兴为您服务，我在听请讲。")  
            text_to_speech("很高兴为您服务，我在听请讲。")  
            
            
            while unknownCount < 2:
                isListenning=True
                user_input = speech_to_text()
                if user_input=='...':
                    continue
                
                display_text(f"You: {user_input}")  
                response = await self.process_query(user_input)
                display_text(f"AI: {response}")  
                
                if await self.getPlayerStatus()=='playing':
                    break
                
                #text_to_speech(f"{response}")  
                
            
            bye_text = os.environ["bye_" + lang]  
            display_text(bye_text) 
            if await self.getPlayerStatus()!='playing':
                text_to_speech(bye_text)  
            
            unknownCount = 0  
            time.sleep(0.1)  

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    finally:
        await client.cleanup()
        print("AI: Bye! See you next time!")

if __name__ == "__main__":
    asyncio.run(main())

#uv run voice.py 启动客户端

  



