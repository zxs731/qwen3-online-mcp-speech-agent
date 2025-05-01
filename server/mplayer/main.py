# server.py
from mcp.server.fastmcp import FastMCP
from typing import Annotated
from pathlib import Path

from pydantic import BaseModel, Field
import json, ast
import pygame  
import requests, json
from io import BytesIO 
import tempfile 
import time
import datetime  
import io 
#import dateutil.parser  
import locale 
import os
#from dotenv import load_dotenv  
import subprocess  


quitReg=False
pause=False
playing=False
# Create an MCP server
mcp = FastMCP("MusicPlayer")
   
@mcp.tool()
def play_music(song_name:str)-> str:
    """
    播放音乐
    
    params: 
        song_name: 歌曲名或关键字
    
    """
    global playing, pause 
    print("playmusic")
    #return f"为您找到歌曲：{song_name} 已开始播放。如果有其他任务请告知我，我先退下了。"
    url='http://music.163.com/api/search/get/web?csrf_token=hlpretag=&hlposttag=&s= %s&type=1&offset=0&total=true&limit=10' % song_name
    res=requests.get(url)
    music_json=json.loads(res.text)
    #print(music_json)
    count=music_json["result"]["songCount"]
    
    if(count>0):
        musicName = downloadAndPlay(music_json,0)
        if musicName:
            print("找到歌曲：'"+musicName+"' 开始播放。请欣赏。")
            #return f"为您找到歌曲：{musicName} 已开始播放。请欣赏。" #"找到歌曲：'"+musicName+"' 开始播放。请欣赏。"
            return {"status":f"歌曲【{musicName}】已开始播放。"} #"找到歌曲：'"+musicName+"' 开始播放。请欣赏。"
        else:
            playing=False
            pause = False
            print("没有找到音乐")
            return "没有找到音乐"
    
    return "没有找到音乐"

def downloadAndPlay(music_json,index):
    global playing, pause 
    count=music_json["result"]["songCount"]
    if index>=count:
        return False
    songid=music_json["result"]["songs"][index]["id"]
    songName=music_json["result"]["songs"][index]["name"]
    url='http://music.163.com/song/media/outer/url?id=%s.mp3' % songid
    response = requests.get(url)  
    audio_data = BytesIO(response.content)  

    temp_file_name = "temp_audio.mp3"  # 临时文件名  
    with open(temp_file_name, 'wb') as temp_file:  
        temp_file.write(audio_data.getbuffer())  
    print(temp_file_name)

    # 初始化pygame  
    pygame.init()  
    try:
        # 播放音乐  
        pygame.mixer.music.load(temp_file_name)  
        pygame.mixer.music.play()
        playing=True
        pause = False
        print(songName)
        return songName
    except Exception as e:  
        print("failed play try next one")
        playing=False
        pause = False
        index+=1
        return downloadAndPlay(music_json,index)

@mcp.tool()       
def isPlaying():
    """
    check if playing
    
    return: 
        Yes if playing, No if not playing
    """
    return playing        

@mcp.tool()      
def stopplay():
    """
    停止播放音乐
    
    返回: 
        播放状态: 已停止
    """
    global playing, pause 
    pygame.mixer.music.stop()  
    playing=False
    pause = False
    return "已停止。"

@mcp.tool()   
def pauseplay():
    """
    暂停音乐播放
    
    返回: 
        播放状态: 已暂停
    """
    global playing, pause
    pygame.mixer.music.pause()
    playing=False
    pause = True
    return "已暂停。"

@mcp.tool()   
def unpauseplay():
    """
    恢复音乐播放
    
    返回：
        播放状态: 已恢复播放
    """
    global playing, pause
    pygame.mixer.music.unpause()
    playing=True
    pause = False
    return "已恢复播放"

    
if __name__ == "__main__":
   print("Server running")
   mcp.run(transport='stdio')


