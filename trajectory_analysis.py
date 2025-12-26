import os
import numpy as np 
import matplotlib.pyplot as plt
import torchaudio
import torch
from sentence_transformers import SentenceTransformer
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import pickle
import soundfile as sf

text_model = SentenceTransformer('all-MiniLM-L6-v2')
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

#----------------------
#loading functions 
#----------------------

#helper functions ------ 
def load_audio(file_path):
        waveform, sample_rate = sf.read(file_path)  
        waveform = torch.tensor(waveform.T, dtype=torch.float32) 
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        return waveform

def embed_speaker(speaker_name):
        speaker_files = sorted([f for f in os.listdir(audio_folder) if f.startswith(speaker_name)])
        speaker_embeddings = []

        for f in speaker_files:
            file_path = os.path.join(audio_folder, f)
            waveform = load_audio(file_path) #^above function to make sure it's ready 
            input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values 
            with torch.no_grad(): #no training, just embedding
                outputs = audio_model(input_values)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
                speaker_embeddings.append(embedding)
        
        return speaker_embeddings 

def load_embed(save_file, save_name):
    """ 
    save_file: the enitre name of the file you want to load eg. conversation_embeddings.pk1
    save_name: the name you want to load it in as eg. embeddings_dict
    returns: the embedding 
    """
    save_path = os.path.join("saved_embeds", save_file)
    with open(save_path, "rb") as f:
        save_name = pickle.load(f)
    print(f"Loaded embeddings from {save_path}")
    return save_name

def embed_audio(root_folder, save_file, speakers):
    """
    root_folder: which scene folder is this for 
    save_file: entire name of file you want to save as
    speakers: array of the speakers in the folder
    """
    audio_model.eval()
    

