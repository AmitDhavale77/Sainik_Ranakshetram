#ProjectGurukul's Voice recorder
#Import necessary modules
import sounddevice as sd
from tkinter import *
import queue
import soundfile as sf
import threading
from tkinter import messagebox
import tkinter as tk
import gc

import os
os.chdir('E:\\amit_sainyaraksh\\')

from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt
import pydub
from pydub.silence import split_on_silence

#pip install SpeechRecognition
import speech_recognition as sr 
import os 

pydub.AudioSegment.ffmpeg = "C:\\Windows\\System32"

import numpy as np


#from google.colab import drive
from IPython.display import display
from IPython.html import widgets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import optim
from torch.nn import functional as F
from transformers import AdamW, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm_notebook
import pandas as pd

sns.set()
train_df = pd.read_csv("E:\\amit_sainyaraksh\\hingtraindataset1.csv").astype(str)
train_df1 = train_df.drop(train_df.columns[[0]],axis = 1)
train_df1.insert(0, 'prefix', 'hing.eng')
train_df1=train_df1.reset_index(drop=True)
train_df1.head()

eval_df1 = pd.read_csv("E:\\amit_sainyaraksh\\hingevaldataset1.csv").astype(str)
eval_df = eval_df1.drop(eval_df1.columns[[0]],axis = 1)
eval_df.insert(0, 'prefix', 'hing.eng')
eval_df=eval_df.reset_index(drop=True)
eval_df.head()

output_file="trial.wav"

model_repo = 'google/mt5-base'
#The model google mt5 base is a Natural Language Processing (NLP) Model implemented in Transformer library,
model_path = 'E:\\amit_sainyaraksh\\mt5_translation27000_2.pt'

max_seq_len = 40
import sentencepiece

tokenizer = AutoTokenizer.from_pretrained(model_repo)
from transformers import AutoConfig
#config = AutoConfig.from_pretrained("E:\\amit_sainyaraksh\\config.json")

model = AutoModelForSeq2SeqLM.from_pretrained('E:\\amit_sainyaraksh\\deepspeech')

model = model.cuda()

model.config.max_length=40  
len(tokenizer.vocab)

train_df1['prefix'].unique()
LANG_TOKEN_MAPPING = {
    'hing.eng': ''
    
}
#A dict which maps hing.eng to ''
special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
special_tokens_dict
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.all_special_tokens
model.config.vocab_size

gc.collect()
torch.cuda.empty_cache()

model.resize_token_embeddings(len(tokenizer))

def encode_str(text, tokenizer, seq_len):
  input_ids = tokenizer.encode(text=text,return_tensors = 'pt',padding = 'max_length',truncation = True,max_length = seq_len)
  return input_ids[0]

gc.collect()
torch.cuda.empty_cache()

save_model=model.load_state_dict(torch.load(model_path))

import pickle
with open('E:\\amit_sainyaraksh\\losses_2.pkl', 'rb') as f:
  losses1 = pickle.load(f)
  
#del model
gc.collect() #Use this method to force the system to try to reclaim the maximum amount of available memory.
torch.cuda.empty_cache()

english_preds1=[]

# create a speech recognition object
r = sr.Recognizer()

def listToString(s): 
    # initialize an empty string
    str1 = ""
     # traverse in the string
    for ele in s:
        str1 += ele 
    # return string
    return str1

# a function that splits the audio file into chunks
# and applies speech recognition
def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    english_preds1=[]
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        wav_file =  pydub.AudioSegment.from_file(file = chunk_filename,format = "wav")
        new_wav_file = wav_file + 10
        chunk_filename1 = os.path.join(folder_name, f"chunk_high{i}.wav")
        new_wav_file.export(out_f =  chunk_filename1,format = "wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename1) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened,language="en-US")
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text1 = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text1
                input_ids = encode_str(text = text1,tokenizer = tokenizer,seq_len = model.config.max_length)
                input_ids = input_ids.unsqueeze(0).cuda()
  #print(input_ids)
                output_tokens = model.generate(input_ids, num_beams=10, num_return_sequences=1, length_penalty = 1, no_repeat_ngram_size=2)
                for token_set in output_tokens:
                  english_preds1.append(tokenizer.decode(token_set, skip_special_tokens=True))
    # return the text for all chunks detected
    x1=listToString(english_preds1)
    #return whole_text
    return x1


def translate_func():
    # Load the file on an object
    data = wavfile.read(output_file)
    # Separete the object elements
    framerate = data[0]
    sounddata = data[1]
    time      = np.arange(0,len(sounddata))/framerate
    # Show information about the object
    ss='Recorded Audio has\n'
    ss=ss+'Sample rate:'+str(framerate)+'Hz'+'\n'
    ss=ss+'Total time:'+str(len(sounddata)/framerate)+'seconds'+'\n'+'Translated Text:'+'\n'
    english_preds1=[]
    textdata=get_large_audio_transcription(output_file)
    # Driver code    
    translate_text.delete(1.0, tk.END)  # Uncomment if you need to replace text instead of adding
    ss=ss+textdata
    translate_text.insert(tk.END, ss)
    return textdata

#Functions to play, stop and record audio in Python voice recorder
#The recording is done as a thread to prevent it being the main process
def threading_rec(x):
   if x == 1:
       #If recording is selected, then the thread is activated
       t1=threading.Thread(target= record_audio)
       t1.start()
   elif x == 2:
       #To stop, set the flag to false
       global recording
       recording = False
       messagebox.showinfo(message="Recording finished")
   elif x == 3:
       #To play a recording, it must exist.
       if file_exists:
           #Read the recording if it exists and play it
           data, fs = sf.read("trial.wav", dtype='float32')
           sd.play(data*10,fs)
           sd.wait()
       else:
           #Display and error if none is found
           messagebox.showerror(message="Record something to play")
   elif x == 4:
        #To play a recording, it must exist.
        if file_exists:
            #Read the recording if it exists and play it
            data, fs = sf.read("trial.wav", dtype='float32')
            sd.play(data*10,fs)
            sd.wait()
            x2=translate_func()
            with open('trial.txt', 'w') as f:
                f.write(x2)
            f.close()
            #translate_text.insert(tk.END, f"Some text\n")
        else:
            #Display and error if none is found
            messagebox.showerror(message="Record something to translate")


#Fit data into queue
def callback(indata, frames, time, status):
   q.put(indata.copy())
   
#Recording function
def record_audio():
   #Declare global variables   
   global recording
   #Set to True to record
   recording= True  
   global file_exists
   #Create a file to save the audio
   messagebox.showinfo(message="Recording Audio. Speak into the mic")
   with sf.SoundFile("trial.wav", mode='w', samplerate=44100,
                       channels=2) as file:
   #Create an input stream to record audio without a preset time
           with sd.InputStream(samplerate=44100, channels=2, callback=callback):
               while recording == True:
                   #Set the variable to True to allow playing the audio later
                   file_exists =True
                   #write into file
                   file.write(q.get())
                   
#Define the user interface for Voice Recorder using Python
voice_rec = Tk()
voice_rec.geometry("1200x800")
voice_rec.title("Sainya Rakshanam Hinglish to English Translator - Developed by Amit Dhavale")
voice_rec.config(bg="#107dc2")
#Create a queue to contain the audio data
q = queue.Queue()
#Declare variables and initialise them
recording = False
file_exists = False

#Label to display app title in Python Voice Recorder Project
title_lbl  = Label(voice_rec, text="Sainya Rakshanam Hinglish to English Translator - Developed by Amit Dhavale", bg="#2176CC",fg='white',relief=RAISED)
title_lbl.grid(row=0, column=0, padx=5, pady=5)
#title_lbl.pack(ipady=5, fill='x')
title_lbl.config(font=("Font", 15))  # change font and size of label

# Username Entry
#mic_frame = Frame(voice_rec, bg="#107dc2")

#Button to record audio
record_btn = Button(voice_rec, width=20, text="Record Audio", bg="yellow",command=lambda m=1:threading_rec(m))
#Stop button
stop_btn = Button(voice_rec,  width=20,text="Stop Recording",bg="red", command=lambda m=2:threading_rec(m))
#Play button
play_btn = Button(voice_rec,  width=20,text="Play Recording", bg="green",command=lambda m=3:threading_rec(m))
#Translate button
trans_btn = Button(voice_rec,  width=20,text="Translate Audio", bg="yellow",command=lambda m=4:threading_rec(m))

#Position buttons
record_btn.grid(row=1,column=0,padx=5, pady=5)
stop_btn.grid(row=1,column=1,padx=5, pady=5)

#Label to display app title in Python Voice Recorder Project
translate_lbl  = Label(voice_rec, text="Translated Text Here...", bg="#107dc2",fg="yellow" )
translate_lbl.grid(row=2, column=0, padx=5, pady=5)
translate_lbl.config(font=("Font", 15))  # change font and size of label
play_btn.grid(row=3,column=0,padx=5, pady=5)
trans_btn.grid(row=3,column=1,padx=5, pady=5)
vertscroll = Scrollbar(orient=VERTICAL,)
translate_text = Text(voice_rec, height = 30, width = 100, wrap = 'word',yscrollcommand=vertscroll.set)
translate_text.grid(row=4, column=0, padx=5, pady=5,sticky="w")
vertscroll.config(command=translate_text.yview, )
translate_text["yscrollcommand"] = vertscroll.set
    
#text_box.grid(column=1, row=4, columnspan=2, sticky="w")
vertscroll.grid(column=1, row=4, sticky="nse")
#vertscroll.config(command=translate_text.yview)
# translate_text.config(yscrollcommand=vertscroll.set)
# vertscroll.grid(column=1, row=4, sticky='NS')

voice_rec.mainloop()












###----------------------------------------------------------------

# import os
# os.chdir('E:\\amit_sainyaraksh\\')
# from scipy.io import wavfile
# from pydub import AudioSegment
# import matplotlib.pyplot as plt

# import pydub
# pydub.AudioSegment.ffmpeg = "C:\\Windows\\System32"

# # assign files
# input_file = "E:\\amit_sainyaraksh\\221201_2042.mp3"
# output_file = "E:\\amit_sainyaraksh\\221201_2042.wav"
# output_file ="E:\\amit_sainyaraksh\\Recording3.wav"
# # import subprocess
# # subprocess.run(["dir"],shell=True)

# # convert mp3 file to wav file
# # sound = AudioSegment.from_mp3(input_file)
# # sound.export(output_file, format="wav")
# # sound = AudioSegment.from_mp3(input_file)
# # sound.export(output_file, format="wav")

# from IPython.display import clear_output
# clear_output()

# import numpy as np
# from IPython.display import Audio
# from scipy.io import wavfile

# # Load the file on an object
# data = wavfile.read(output_file)

# # Separete the object elements
# framerate = data[0]
# sounddata = data[1]
# time      = np.arange(0,len(sounddata))/framerate

# # Show information about the object
# print('Sample rate:',framerate,'Hz')
# print('Total time:',len(sounddata)/framerate,'s')

# # import librosa
# # x, sr = librosa.load(output_file)
# # print(x.shape)
# # print(sr)

# # import matplotlib.pyplot as plt
# # import librosa.display
# # plt.figure(figsize=(14, 5))
# # librosa.display.waveshow(x, sr=sr)

# # import IPython
# # from scipy.io import wavfile
# # import noisereduce as nr
# # import soundfile as sf
# # from noisereduce.generate_noise import band_limited_noise
# # import matplotlib.pyplot as plt
# # import urllib.request
# # import numpy as np
# # import io
# # from scipy.io import wavfile

# # reduced_noise = nr.reduce_noise(y = x[:], sr=sr, thresh_n_mult_nonstationary=1.5,stationary=False)
# # print(reduced_noise.shape)
# # print(sr)

# # import matplotlib.pyplot as plt
# # import librosa.display
# # plt.figure(figsize=(14, 5))
# # librosa.display.waveshow(reduced_noise, sr=sr)

# #from google.colab import drive
# from IPython.display import display
# from IPython.html import widgets
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# import torch
# from torch import optim
# from torch.nn import functional as F
# from transformers import AdamW, AutoModelForSeq2SeqLM, AutoTokenizer
# from transformers import get_linear_schedule_with_warmup
# from tqdm import tqdm_notebook
# import pandas as pd

# sns.set()

# train_df = pd.read_csv("E:\\amit_sainyaraksh\\hingtraindataset1.csv").astype(str)
# train_df1 = train_df.drop(train_df.columns[[0]],axis = 1)
# train_df1.insert(0, 'prefix', 'hing.eng')
# train_df1=train_df1.reset_index(drop=True)
# train_df1.head()

# eval_df1 = pd.read_csv("E:\\amit_sainyaraksh\\hingevaldataset1.csv").astype(str)
# eval_df = eval_df1.drop(eval_df1.columns[[0]],axis = 1)
# eval_df.insert(0, 'prefix', 'hing.eng')
# eval_df=eval_df.reset_index(drop=True)
# eval_df.head()

# model_repo = 'google/mt5-base'
# #The model google mt5 base is a Natural Language Processing (NLP) Model implemented in Transformer library,
# model_path = 'E:\\amit_sainyaraksh\\mt5_translation27000_2.pt'

# max_seq_len = 40
# import sentencepiece

# tokenizer = AutoTokenizer.from_pretrained(model_repo)
# from transformers import AutoConfig
# #config = AutoConfig.from_pretrained("E:\\amit_sainyaraksh\\config.json")

# model = AutoModelForSeq2SeqLM.from_pretrained('E:\\amit_sainyaraksh\\deepspeech')

# model = model.cuda()

# model.config.max_length=40  
# len(tokenizer.vocab)

# train_df1['prefix'].unique()
# LANG_TOKEN_MAPPING = {
#     'hing.eng': ''
    
# }
# #A dict which maps hing.eng to ''
# special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
# special_tokens_dict
# tokenizer.add_special_tokens(special_tokens_dict)
# tokenizer.all_special_tokens
# model.config.vocab_size

# import gc

# gc.collect()

# torch.cuda.empty_cache()

# model.resize_token_embeddings(len(tokenizer))

# def encode_str(text, tokenizer, seq_len):
#   input_ids = tokenizer.encode(text=text,return_tensors = 'pt',padding = 'max_length',truncation = True,max_length = seq_len)
#   return input_ids[0]

# #map_location=torch.device('cpu')

# import gc

# gc.collect()

# torch.cuda.empty_cache()

# save_model=model.load_state_dict(torch.load(model_path))

# import pickle
# with open('E:\\amit_sainyaraksh\\losses_2.pkl', 'rb') as f:

#   losses1 = pickle.load(f)
  
# import gc
# #del model
# gc.collect() #Use this method to force the system to try to reclaim the maximum amount of available memory.

# torch.cuda.empty_cache()

# # window_size = 50
# # smoothed_losses = []
# # for i in range(len(losses1)-window_size):
# #   smoothed_losses.append(np.mean(losses1[i:i+window_size]))#plot mean of each windowed size loss

# # plt.plot(smoothed_losses[100:])

# import sacrebleu
# english_truth = [eval_df.loc[eval_df["prefix"] == "hing.eng"]["English"].tolist()]
# english_truth = english_truth[0][:500]
# to_english=eval_df.loc[eval_df["prefix"] == "hing.eng"]["English"].tolist()
# to_english=to_english[:500]
# english_preds=[]

# for i in to_english:
#   input_ids = encode_str(
#     text = i,
#     tokenizer = tokenizer,
#     seq_len = model.config.max_length)
#   input_ids = input_ids.unsqueeze(0).cuda()
#   #print(input_ids)
#   output_tokens = model.generate(input_ids, num_beams=10, num_return_sequences=1, length_penalty = 1, no_repeat_ngram_size=2)
#   for token_set in output_tokens:
#     english_preds.append(tokenizer.decode(token_set, skip_special_tokens=True))
# # hing_eng_bleu = sacrebleu.corpus_bleu(english_preds, [english_truth])
# # print("--------------------------")
# # print("Hinglish to English: ", hing_eng_bleu.score) #it tells me by how much my overall predicted texts are similer to input texts
# #english_preds2=[]

# print(english_preds)



# import os 
# import pydub
# from pydub.playback import play
# from pydub import AudioSegment
# from pydub.silence import split_on_silence
# import scipy

# #pip install SpeechRecognition
# import speech_recognition as sr 
# import os 
# from pydub import AudioSegment
# from pydub.silence import split_on_silence

# # create a speech recognition object
# r = sr.Recognizer()

# # a function that splits the audio file into chunks
# # and applies speech recognition
# def get_large_audio_transcription(path):
#     """
#     Splitting the large audio file into chunks
#     and apply speech recognition on each of these chunks
#     """
#     # open the audio file using pydub
#     sound = AudioSegment.from_wav(path)  
#     # split audio sound where silence is 700 miliseconds or more and get chunks
#     chunks = split_on_silence(sound,
#         # experiment with this value for your target audio file
#         min_silence_len = 500,
#         # adjust this per requirement
#         silence_thresh = sound.dBFS-14,
#         # keep the silence for 1 second, adjustable as well
#         keep_silence=500,
#     )
#     folder_name = "audio-chunks4"
#     # create a directory to store the audio chunks
#     if not os.path.isdir(folder_name):
#         os.mkdir(folder_name)
#     whole_text = ""
#     # process each chunk 
#     for i, audio_chunk in enumerate(chunks, start=1):
#         # export audio chunk and save it in
#         # the `folder_name` directory.
#         chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
#         audio_chunk.export(chunk_filename, format="wav")
#         wav_file =  pydub.AudioSegment.from_file(file = chunk_filename,format = "wav")
#         new_wav_file = wav_file + 10
#         chunk_filename1 = os.path.join(folder_name, f"chunk_high{i}.wav")
#         new_wav_file.export(out_f =  chunk_filename1,format = "wav")
#         # recognize the chunk
#         with sr.AudioFile(chunk_filename1) as source:
#             audio_listened = r.record(source)
#             # try converting it to text
#             try:
#                 text = r.recognize_google(audio_listened,language="en-US")
#             except sr.UnknownValueError as e:
#                 print("Error:", str(e))
#             else:
#                 text1 = f"{text.capitalize()}. "
#                 print(chunk_filename, ":", text)
#                 whole_text += text1
#                 input_ids = encode_str(text = text1,tokenizer = tokenizer,seq_len = model.config.max_length)
#                 input_ids = input_ids.unsqueeze(0).cuda()
#   #print(input_ids)
#                 output_tokens = model.generate(input_ids, num_beams=10, num_return_sequences=1, length_penalty = 1, no_repeat_ngram_size=2)
#                 for token_set in output_tokens:
#                   english_preds1.append(tokenizer.decode(token_set, skip_special_tokens=True))
#     # return the text for all chunks detected
#     return whole_text

# english_preds1=[]
# textdata=get_large_audio_transcription(output_file)
# textdata

# def listToString(s):
 
#     # initialize an empty string
#     str1 = ""
 
#     # traverse in the string
#     for ele in s:
#         str1 += ele
 
#     # return string
#     return str1
 
 
# # Driver code
# x1=listToString(english_preds1)
# x1





