import os
import pickle
import torch
import umap
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import torchaudio
import torch.nn.functional as F
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
    """
    reads wav using soundfile to convert to a tensor
    """
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
    """
    uses wav2vec to convert raw audio (in tensor form) to embedding (vector representing tone, prosody)
    """
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
    """
    uses miniLM to encode list of sentences into embeddings 
    """
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

    loops through speakers audio files in a folder, returns dict of audio emb
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
    returns dict of text embed of dilaogue
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
    """
    concatenates audio embeds and text embeds.... maybe don't want this tbh  
    """
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
    """
    path should be like the whole path.. i think like eg. saved_embeds/audio_embeddings.pkl
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_embeddings(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_annotations_txt(file_path, speakers):
    annotations = {sp.lower(): {} for sp in speakers}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sp_utt, text = line.split(":", 1)
                sp, utt = sp_utt.split(",", 1)
                sp = sp.strip().lower()
                utt = int(utt.strip()) - 1  # convert 1-based to 0-based
                annotations[sp][utt] = text.strip()
            except ValueError:
                print(f"Skipping malformed line: {line}")
    
    return annotations


#----------------------
#pca/reduction functions
#----------------------
def reduce_embeddings(X, method="pca", n_components=2):
    """
    X is the 2d mtrix of embeddings, torch stack basically of embeddings_dict[speaker1]
    """
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
    plt.show()

def plot_all_layers(emotional_dict, scene_name="Scene"):
    """
    for all the layers, plot metric for each layer ^
    """
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
    for i in range(len(h_A)):
        sims.append(cosine_similarity(h_A[i], h_B[i]))
    return sims

def cross_speaker_delta(deltas_A, deltas_B):
    """
    compares the DELTAS. checks if B's change ALSO FOLLOWS A's change. 
    each delta is above^ func. how does A change from one utterance i to i+1
    deltaB[i] is HOW B changed in response. SO !!!!! speaker A has to be the first initiator 
    and speaker B is the second.. 
    """
    dirs = []
    for i in range(len(deltas_A)):
        dirs.append(cosine_similarity(deltas_A[i], deltas_B[i]))
    return dirs

def amplification(deltas_A, deltas_B):
    """
    salience. who is "more salient" norm is the euclidean length(could change later) 
    """
    amps = []
    for i in range(len(deltas_A)):
        norm_A = torch.norm(deltas_A[i])
        norm_B = torch.norm(deltas_B[i])
        amps.append((norm_B / (norm_A + 1e-6)).item())
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
#more plots! another plot -> w annotations for interpretation
#----------------------
def plot_emotional_layers(emotional_dict, speakers, scene_name="Scene", annotations=None):
    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=False)

    for sp in speakers:
        deltas = [torch.norm(d).item() for d in emotional_dict["layer1_deltas"][sp]]
        axs[0].plot(range(1, len(deltas)+1), deltas, marker="o", label=sp)
    axs[0].set_title("Self change magnitude")
    axs[0].set_ylabel("||Δ||")
    axs[0].legend()
    x_labels = [f"{i}-{i+1}" for i in range(1, len(deltas)+1)]
    axs[0].set_xticks(range(1, len(deltas)+1))
    axs[0].set_xticklabels(x_labels, rotation=45)
    axs[0].set_xlabel("Utterance pair")

    sims = emotional_dict["layer2_alignment_AB"]
    axs[1].plot(range(1, len(sims)+1), sims, marker="o", color="purple")
    axs[1].set_title("Inter-speaker alignment")
    axs[1].set_ylabel("cos(hA, hB)")
    x_labels = [str(i) for i in range(1, len(sims)+1)]
    axs[1].set_xticks(range(1, len(sims)+1))
    axs[1].set_xticklabels(x_labels, rotation=0)
    axs[1].set_xlabel("Utterance index")

    contagion = emotional_dict["layer3_contagion_AB"]
    axs[2].plot(range(1, len(contagion)+1), contagion, marker="o", color="orange")
    axs[2].set_title("Contagion")
    axs[2].set_ylabel("cos(ΔA, ΔB)")
    x_labels = [f"{i}-{i+1}" for i in range(1, len(contagion)+1)]
    axs[2].set_xticks(range(1, len(contagion)+1))
    axs[2].set_xticklabels(x_labels, rotation=45)
    axs[2].set_xlabel("Utterance pair")

    amp = emotional_dict["layer4_amplification_AB"]
    axs[3].plot(range(1, len(amp)+1), amp, marker="o", color="green")
    axs[3].axhline(1.0, linestyle="--", alpha=0.5)
    axs[3].set_title("Amplification")
    axs[3].set_ylabel("||ΔB|| / ||ΔA||")
    x_labels = [f"{i}-{i+1}" for i in range(1, len(amp)+1)]
    axs[3].set_xticks(range(1, len(amp)+1))
    axs[3].set_xticklabels(x_labels, rotation=45)
    axs[3].set_xlabel("Utterance pair")

    for sp in speakers:
        inertia = emotional_dict["layer5_inertia"][sp]
        axs[4].plot(range(1, len(inertia)+1), inertia, marker="o", label=sp)
    axs[4].set_title("Emotional inertia")
    axs[4].set_ylabel("cos(h[t], h[t+1])")
    axs[4].legend()
    x_labels = [f"{i}-{i+1}" for i in range(1, len(inertia)+1)]
    axs[4].set_xticks(range(1, len(inertia)+1))
    axs[4].set_xticklabels(x_labels, rotation=45)
    axs[4].set_xlabel("Utterance pair")

    if annotations:
        for sp, sp_ann in annotations.items():
            for idx, text in sp_ann.items():
                x_pos = idx + 1
                if x_pos <= len(axs[0].lines[0].get_xdata()):
                    axs[0].axvline(x_pos, color="red", linestyle="--", alpha=0.5)
                    axs[0].text(
                        x_pos + 0.1,
                        axs[0].get_ylim()[1]*0.9,
                        f"{sp}: {text}",
                        color="red",
                        rotation=90,
                        fontsize=8
                    )

    plt.suptitle(scene_name)
    plt.tight_layout()
    plt.show()

def plot_delta_directions(
    embeddings_dict,
    speakers,
    reducer="pca",
    n_components=2,
    title="Directional Emotional Change"
):
    """
    plots directions of delta 
    """

    plt.figure(figsize=(10, 6))

    # assign colors dynamically
    cmap = plt.get_cmap("tab10")
    speaker_colors = {s: cmap(i) for i, s in enumerate(speakers)}

    for speaker in speakers:
        emb_list = embeddings_dict[speaker]

        X = torch.stack(emb_list).numpy()

        X_red = reduce_embeddings(X, method=reducer, n_components=n_components)

        plt.scatter(
            X_red[:, 0],
            X_red[:, 1],
            color=speaker_colors[speaker],
            label=speaker
        )

        for i in range(1, len(X_red)):
            dx = X_red[i, 0] - X_red[i - 1, 0]
            dy = X_red[i, 1] - X_red[i - 1, 1]

            plt.arrow(
                X_red[i - 1, 0],
                X_red[i - 1, 1],
                dx,
                dy,
                head_width=0.02,
                alpha=0.7,
                length_includes_head=True,
                color=speaker_colors[speaker]
            )

            plt.text(
                X_red[i, 0],
                X_red[i, 1],
                str(i + 1),
                fontsize=9
            )

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def summarize_speaker_metrics(emotional_dict, speaker):
    """
    summarizes for single speaker
    """

    deltas = emotional_dict["layer1_deltas"][speaker]
    inertia = emotional_dict["layer5_inertia"][speaker]

    #how much did emotional state shift disregard direction
    delta_mags = [torch.norm(d).item() for d in deltas]

    return {
        "mean_delta": float(np.mean(delta_mags)) if delta_mags else 0.0,
        "max_delta": float(np.max(delta_mags)) if delta_mags else 0.0,
        "mean_inertia": float(np.mean(inertia)) if inertia else 0.0,
    }

def summarize_cross_metrics(emotional_dict):
    return {
        "mean_alignment": float(np.mean(emotional_dict["layer2_alignment_AB"])),
        "mean_contagion": float(np.mean(emotional_dict["layer3_contagion_AB"])),
        "mean_amplification": float(np.mean(emotional_dict["layer4_amplification_AB"]))
    }

def flag_trends(emotional_dict, speaker, delta_thresh=0.6, inertia_thresh=0.75, amp_thresh=1.1,
                align_thresh=0.6, contagion_thresh=0.6):
    """
    Returns a dictionary with per-utterance flags for trends in metrics.
    """
    trends = {
        "volatility": [],
        "reactive": [],
        "accommodating": [],
        "rigid": [],
        "notable_shift": []
    }

    deltas = [torch.norm(d).item() for d in emotional_dict["layer1_deltas"][speaker]]
    inertia = emotional_dict["layer5_inertia"][speaker]

    amplifications = emotional_dict["layer4_amplification_AB"]
    alignments = emotional_dict["layer2_alignment_AB"]
    contagions = emotional_dict["layer3_contagion_AB"]

    n = len(deltas)

    for i in range(n):
        if deltas[i] > delta_thresh and inertia[i] < inertia_thresh:
            trends["volatility"].append(i)

        if i < len(amplifications) and amplifications[i] > amp_thresh and inertia[i] < inertia_thresh:
            trends["reactive"].append(i)

        if i < len(alignments) and alignments[i] > align_thresh and i < len(contagions) and contagions[i] > contagion_thresh:
            trends["accommodating"].append(i)

        if i < len(amplifications) and i < len(contagions) and inertia[i] > inertia_thresh \
           and amplifications[i] < amp_thresh and contagions[i] < contagion_thresh:
            trends["rigid"].append(i)

        if deltas[i] > delta_thresh:
            trends["notable_shift"].append(i)

    return trends

def interpret_speaker(
    emotional_dict,
    speaker,
    delta_thresh=0.6,
    inertia_thresh=0.75,
    amp_thresh=1.1,
    align_thresh=0.6,
    contagion_thresh=0.6,
    verbose=True
):
    """
    Returns multiple descriptors of speaker behavior based on emotional metrics.
    Includes reasoning if verbose=True.
    """
    summary = summarize_speaker_metrics(emotional_dict, speaker)
    cross = summarize_cross_metrics(emotional_dict)

    descriptors = []
    reasoning = []

    if summary["mean_inertia"] > inertia_thresh and summary["mean_delta"] < delta_thresh:
        descriptors.append("stable")
        reasoning.append(f"mean_inertia={summary['mean_inertia']:.2f} > {inertia_thresh}, mean_delta={summary['mean_delta']:.2f} < {delta_thresh}")
    if summary["mean_delta"] > delta_thresh and summary["mean_inertia"] < inertia_thresh:
        descriptors.append("volatile")
        reasoning.append(f"mean_delta={summary['mean_delta']:.2f} > {delta_thresh}, mean_inertia={summary['mean_inertia']:.2f} < {inertia_thresh}")

    if cross["mean_amplification"] > amp_thresh and summary["mean_inertia"] < inertia_thresh:
        descriptors.append("reactive")
        reasoning.append(f"mean_amplification={cross['mean_amplification']:.2f} > {amp_thresh}, mean_inertia={summary['mean_inertia']:.2f} < {inertia_thresh}")

    if cross["mean_alignment"] > align_thresh and cross["mean_contagion"] > contagion_thresh:
        descriptors.append("accommodating")
        reasoning.append(f"mean_alignment={cross['mean_alignment']:.2f} > {align_thresh}, mean_contagion={cross['mean_contagion']:.2f} > {contagion_thresh}")

    if summary["mean_inertia"] > inertia_thresh and cross["mean_amplification"] < amp_thresh and cross["mean_contagion"] < contagion_thresh:
        descriptors.append("rigid")
        reasoning.append(f"mean_inertia={summary['mean_inertia']:.2f} > {inertia_thresh}, mean_amplification={cross['mean_amplification']:.2f} < {amp_thresh}, mean_contagion={cross['mean_contagion']:.2f} < {contagion_thresh}")

    if summary["max_delta"] > delta_thresh:
        descriptors.append("experiences noticeable emotional shifts")
        reasoning.append(f"max_delta={summary['max_delta']:.2f} > {delta_thresh}")

    if not descriptors:
        descriptors.append("no strong emotional pattern")
        reasoning.append(f"All metrics below thresholds")

    if verbose:
        return f"{speaker}: {', '.join(descriptors)}.", "\n".join(reasoning)
    else:
        return f"{speaker}: {', '.join(descriptors)}."

def interpret_speaker_with_trends(emotional_dict, speaker, delta_thresh=0.6, inertia_thresh=0.75,
                                  amp_thresh=1.1, align_thresh=0.6, contagion_thresh=0.6):
    """
    Returns overall descriptors + per-utterance trends for a speaker.
    """
    overall, reasoning = interpret_speaker(
        emotional_dict,
        speaker,
        delta_thresh,
        inertia_thresh,
        amp_thresh,
        align_thresh,
        contagion_thresh,
        verbose=True
    )

    trends = flag_trends(
        emotional_dict,
        speaker,
        delta_thresh,
        inertia_thresh,
        amp_thresh,
        align_thresh,
        contagion_thresh
    )

    return {
        "summary": overall,
        "reasoning": reasoning,
        "trends": trends
    }

def compute_scene_thresholds(
    emotional_dict,
    speakers,
    delta_q=0.7,
    inertia_q=0.7,
    amp_q=0.7,
    align_q=0.7,
    contagion_q=0.7
):
    """
    scene-relative thresholds
    """

    all_mean_deltas = []
    all_mean_inertias = []

    for speaker in speakers:
        summary = summarize_speaker_metrics(emotional_dict, speaker)
        all_mean_deltas.append(summary["mean_delta"])
        all_mean_inertias.append(summary["mean_inertia"])

    cross = summarize_cross_metrics(emotional_dict)

    thresholds = {
        "delta_thresh": float(np.quantile(all_mean_deltas, delta_q)),
        "inertia_thresh": float(np.quantile(all_mean_inertias, inertia_q)),
        "amp_thresh": float(cross["mean_amplification"]),   # single value → relative use
        "align_thresh": float(cross["mean_alignment"]),
        "contagion_thresh": float(cross["mean_contagion"]),
    }

    return thresholds

def generate_scene_summary_with_trends(emotional_dict, speakers):
    thresholds = compute_scene_thresholds(emotional_dict, speakers)
    summaries = {}

    for speaker in speakers:
        result = interpret_speaker_with_trends(
            emotional_dict,
            speaker,
            **thresholds
        )
        print(result["summary"])
        print("Reasoning:")
        print(result["reasoning"])
        print("Per-utterance trends:")
        for key, indices in result["trends"].items():
            if indices:
                print(f"  {key}: utterances {', '.join(str(i+1) for i in indices)}")
        print("-"*50)
        summaries[speaker] = result

    return summaries

def plot_emotional_stability_map(emotional_dict, speakers):
    """
    stability 
    top-left: stable
    bottom-right: volatile
    top-right: intense, consistent 
    bottom-left: muted
    """
    plt.figure(figsize=(6,6))

    for speaker in speakers:
        summary = summarize_speaker_metrics(emotional_dict, speaker)
        x = summary["mean_delta"]
        y = summary["mean_inertia"]

        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, speaker)

    plt.xlabel("Mean Delta (Volatility)")
    plt.ylabel("Mean Inertia (Stability)")
    plt.title("Emotional Stability Map (Scene-relative)")
    plt.grid(True)
    plt.show()







