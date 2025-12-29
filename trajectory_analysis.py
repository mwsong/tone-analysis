import os
import pickle
import torch
import umap
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torchaudio
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA


from sentence_transformers import SentenceTransformer
from transformers import Wav2Vec2Processor, Wav2Vec2Model


#----------------------
#models (loaded)
#----------------------
text_model = SentenceTransformer('all-MiniLM-L6-v2')
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
audio_model.eval()

#----------------------
#loading functions 
#----------------------

#helper functions ------ 
def load_audio(file_path):
    waveform, sample_rate = sf.read(file_path)
    waveform = torch.tensor(waveform, dtype=torch.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        )(waveform)

    return waveform


def get_audio_embedding(waveform):
    inputs = processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values

    with torch.no_grad():
        outputs = audio_model(inputs)
        emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

    return emb


def get_text_embeddings(utterances):
    return text_model.encode(
        utterances,
        convert_to_tensor=True
    )


def normalize(emb):
    return F.normalize(emb, dim=0)


#embedding functions 
def embed_audio_scene(audio_folder, speakers):
    """
    audio_folder: folder containing wavs
    speakers: ["andrew", "justine"]
    """
    embeddings = {s: [] for s in speakers}

    for speaker in speakers:
        files = sorted(f for f in os.listdir(audio_folder) if f.startswith(speaker))

        for fname in files:
            path = os.path.join(audio_folder, fname)
            waveform = load_audio(path)
            emb = get_audio_embedding(waveform)
            embeddings[speaker].append(emb)

    return embeddings


def embed_text_scene(transcript_path, speakers):
    """
    transcript format:
    ANDREW blah blah
    JUSTINE blah blah
    """
    text_dict = {s: [] for s in speakers}

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            for speaker in speakers:
                if line.upper().startswith(speaker.upper()):
                    text = line[len(speaker):].strip()
                    text_dict[speaker].append(text)

    text_embeddings = {}
    for speaker in speakers:
        text_embeddings[speaker] = get_text_embeddings(text_dict[speaker])

    return text_embeddings


def combine_embeddings(audio_embs, text_embs, normalize_first=True):
    combined = {}

    for speaker in audio_embs:
        combined[speaker] = []

        for a, t in zip(audio_embs[speaker], text_embs[speaker]):
            if normalize_first:
                a = normalize(a)
                t = normalize(t)

            combined[speaker].append(torch.cat([a, t]))

    return combined

#saving/load embeds 
def save_embeddings(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)

#----------------------
#pca/reduction functions
#----------------------
def reduce_embeddings(X, method="pca", n_components=2):
    if method is None:
        return X

    if method == "pca":
        reducer = PCA(n_components=n_components)
        return reducer.fit_transform(X)

    if method == "umap":
        reducer = umap.UMAP(n_components=n_components)
        return reducer.fit_transform(X)

    raise ValueError(f"Unknown reduction method: {method}")

#----------------------
#plotting functions
#----------------------
def plot_embedding_trajectory(
    embeddings_dict,
    reducer="pca",
    n_components=2,
    annotate=True,
    show_deltas=False,
    title="Embedding Trajectory"
):
    plt.figure(figsize=(10, 6))
    colors = {"andrew": "blue", "justine": "red"}

    for speaker, emb_list in embeddings_dict.items():
        X = torch.stack(emb_list).numpy()

        # reduce
        X_red = reduce_embeddings(X, method=reducer, n_components=n_components)

        # plot path
        plt.plot(
            X_red[:, 0],
            X_red[:, 1],
            marker="o",
            label=speaker,
            color=colors.get(speaker, "black")
        )

        # annotate time
        if annotate:
            for i, (x, y) in enumerate(X_red):
                plt.text(x, y, str(i+1), fontsize=9)

        # arrows + deltas
        if show_deltas:
            deltas = compute_self_deltas(emb_list)
            for i in range(1, len(X_red)):
                color = "red" if deltas[i-1] < 0.95 else colors.get(speaker, "black")
                plt.arrow(
                    X_red[i-1, 0], X_red[i-1, 1],
                    X_red[i, 0] - X_red[i-1, 0],
                    X_red[i, 1] - X_red[i-1, 1],
                    head_width=0.02,
                    alpha=0.6,
                    color=color,
                    length_includes_head=True
                )

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    

def plot_metric(metric_list, speaker_name, scene_name="Scene", ylabel="Value"):
    """
    Plot by time for the different emotional analysis layers below 
    """
    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(metric_list)+1), metric_list, marker='o', label=speaker_name)
    plt.title(f"{scene_name}: {speaker_name}")
    plt.xlabel("Utterance #")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_all_layers(emotional_dict, scene_name="Scene"):
    for layer_name, data in emotional_dict.items():
        if isinstance(data, dict):
            for speaker, metric_list in data.items():
                # If it's deltas (list of tensors), convert to norms first
                if layer_name in ["layer1_deltas"]:
                    metric_list = [torch.norm(d).item() for d in metric_list]
                
                plot_metric(
                    metric_list,
                    speaker_name=speaker,
                    scene_name=scene_name,
                    ylabel=layer_name
                )
        else:
            plot_metric(
                data,
                speaker_name=layer_name,
                scene_name=scene_name,
                ylabel=layer_name
            )



#----------------------
#emotional trajectory analysis layers: functions !!
#!! BTW make sure B is the SECOND speaker in the dialogue (B is reacting to A)
#h_A is the list of tensors for one speaker embedding eg. embeddings_dict["Andrew"]
#deltas_A is what's returned by inter-speaker alignment
#----------------------
def cosine_similarity(a, b):
    """
    a & b are both single vectors (tensors) 
    computes cosinge sim along feature dimension dim= 1
    then returns the python float
    """
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()

def compute_self_deltas(embeddings):
    """
    Goal: how does speaker embedding change over time 
    Say speaker A
    deltaA = h[t] - h[t-1]
    It's basically all the vector differences between consecutive embeddings (per speaker)
    """
    deltas = [embeddings[i] - embeddings[i-1] for i in range(1, len(embeddings))]
    return deltas

def inter_speaker_alignment(h_A, h_B):
    """
    This is like above but between speakers. like is speaker B reacting to speaker A
    so similarity_AB[i] = cos(A[i], B[i])
    A and B or h_A and h_B is the list of embeddings for speakers 
    does B's embedding follow A's embedding? 
    high sim => B's state aligns with A's. returns list of sim scores OVER TIME
    """
    sims = []
    for i in range(len(h_A)-1):
        sims.append(cosine_similarity(h_A[i], h_B[i]))
    return sims

def cross_speaker_delta(deltas_A, deltas_B):
    """
    compares the DELTAS. checks if B's change ALSO FOLLOWS A's change. 
    each delta is above^ func. how does A change from one utterance i to i+1
    deltaB[i+1] is HOW B changed in response. SO !!!!! speaker A has to be the first initiator 
    and speaker B is the second.. 
    """
    dirs = []
    for i in range(len(deltas_A)-1):
        dirs.append(cosine_similarity(deltas_A[i], deltas_B[i]))
    return dirs

def amplification(deltas_A, deltas_B):
    """
    whose emotional change is stronger/weaker. norm is the euclidean length(could change later)
    and normB/normA is the amplification ratio 
    """
    amps = []
    for i in range(len(deltas_A)-1):
        norm_A = torch.norm(deltas_A[i])
        norm_B = torch.norm(deltas_B[i])
        amps.append((norm_B / norm_A).item())
    return amps

def autocorr(embeddings):
    """
    how stable is a speakers emotional state OVER time? 
    cosine sim of consecutive embeddings h[i] to h[i+1]
    if high -> likely emotional consistency
    """
    acorr = []
    for i in range(len(embeddings)-1):
        acorr.append(cosine_similarity(embeddings[i], embeddings[i+1]))
    return acorr


def run_emotional_layers(embeddings_dict, speaker_A, speaker_B):

    h_A = embeddings_dict[speaker_A]
    h_B = embeddings_dict[speaker_B]

    # intra-speaker
    deltas_A = compute_self_deltas(h_A)
    deltas_B = compute_self_deltas(h_B)

    # inter-speaker alignment
    sim_AB = inter_speaker_alignment(h_A, h_B)

    # contagion
    dir_AB = cross_speaker_delta(deltas_A, deltas_B)

    # amplification
    amp_AB = amplification(deltas_A, deltas_B)

    # emotional inertia
    inertia_A = autocorr(h_A)
    inertia_B = autocorr(h_B)

    return {
        "layer1_deltas": {
            speaker_A: deltas_A,
            speaker_B: deltas_B
        },
        "layer2_alignment_AB": sim_AB,
        "layer3_contagion_AB": dir_AB,
        "layer4_amplification_AB": amp_AB,
        "layer5_inertia": {
            speaker_A: inertia_A,
            speaker_B: inertia_B
        }
    }


#----------------------
#another plot -> w annotations for interpretation
#----------------------
def plot_emotional_layers(h_A, h_B, speaker_A_name="Andrew", speaker_B_name="Justine",
                          annotations=None):
    """
    Plots all 5 emotional layers for two speakers in a dialogue.
    annotations: dict {utterance_index: label} to overlay points of interest
    """
    # Compute metrics
    deltas_A = compute_self_deltas(h_A)
    deltas_B = compute_self_deltas(h_B)
    
    self_delta_A = [torch.norm(d).item() for d in deltas_A]
    self_delta_B = [torch.norm(d).item() for d in deltas_B]
    
    alignment = inter_speaker_alignment(h_A, h_B)
    contagion = cross_speaker_delta(deltas_A, deltas_B)
    amp = amplification(deltas_A, deltas_B)
    inertia_A = autocorr(h_A)
    inertia_B = autocorr(h_B)
    
    utterance_nums = list(range(2, len(h_A)+1))  
    
    fig, axs = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    
    # Layer 1: Self-deltas
    axs[0].plot(utterance_nums, self_delta_A, marker='o', label=f"{speaker_A_name} Δ")
    axs[0].plot(utterance_nums, self_delta_B, marker='o', label=f"{speaker_B_name} Δ")
    axs[0].set_ylabel("Self-delta magnitude")
    axs[0].set_title("Layer 1: Emotional change magnitude (Self-deltas)")
    axs[0].legend()
    
    # Layer 2: Inter-speaker alignment
    axs[1].plot(utterance_nums, alignment, marker='o', color='purple')
    axs[1].set_ylabel("Cosine similarity")
    axs[1].set_title("Layer 2: Inter-speaker alignment (Tone similarity)")
    
    # Layer 3: Cross-speaker delta (contagion)
    axs[2].plot(utterance_nums[:-1], contagion, marker='o', color='orange')
    axs[2].set_ylabel("Cosine similarity")
    axs[2].set_title("Layer 3: Cross-speaker delta (Contagion / mirrored change)")
    
    # Layer 4: Amplification
    axs[3].plot(utterance_nums[:-1], amp, marker='o', color='green')
    axs[3].axhline(1, color='gray', linestyle='--', label='Equal Δ magnitude')
    axs[3].set_ylabel("Amplification ratio (B/A)")
    axs[3].set_title("Layer 4: Amplification (relative magnitude)")
    axs[3].legend()
    
    # Layer 5: Autocorrelation (inertia)
    axs[4].plot(utterance_nums, inertia_A, marker='o', label=f"{speaker_A_name} inertia")
    axs[4].plot(utterance_nums, inertia_B, marker='o', label=f"{speaker_B_name} inertia")
    axs[4].set_ylabel("Cosine similarity")
    axs[4].set_title("Layer 5: Emotional inertia (stability over time)")
    axs[4].set_xlabel("Utterance #")
    axs[4].legend()
    
    # Overlay annotations if provided
    if annotations:
        for ax in axs:
            for ut_num, label in annotations.items():
                ax.axvline(ut_num, color='red', linestyle='--', alpha=0.5)
                ax.text(ut_num + 0.1, ax.get_ylim()[1]*0.9, label, color='red', rotation=90)
    
    plt.tight_layout()
    plt.show()
