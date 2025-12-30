# Exploratory Analysis of Emotional Trajectories with Wav2Vec

# Overview 
My project analyzes heated scenes from some movies I like to track emotional and prosodic dynamics over time. I wanted to mathematically quantify how, in conversations, peoples' emotions evolve, respond to each other, mirror each other, and escalate throughout. 

# Motivation 
When we're in dialogue, we perform appraisal and try to gauge the emotional space we're in, the interpersonal safety, and the abrupt changes of the other performer. My main questions and drivers towards this exploration were:
1. During appraisal, especially in moments of tension and stress, how do people form an idea of the emotional space?
2. How do we then externalize our intuition into a more formal system?
3. Is there some degree of "objectiveness" in how we speak that we didn't have the tools to realize before, but now we can with transformers?

# Methods 
Audio embeddings: Wav2Vec2-mini 
Text embeddings(obsolete): miniLM (I ended up deciding that I wasn't sure how to concatenate the two embeddings together. If text embeddings were encoding on a different basis than the Wav2Vec, which captures prosody and tone, then I felt it could add noise to the information) 

Layers of Analysis:
1. mean value of raw embeddings 
2. self change, how much change is happening from one response to the next within one speaker. (delta = h[t] - h[t-1]) just subtracting the vectors
3. inter-speaker alignment. do the two speakers agree in the emotional space? (cosine similarity of h_A[t] and h_B[t] where h_B[t] would be the response to h_A[t])
4. speaker contagion. is the speakers change similar to the other speakers change? or is there some influence in the direction of their prosodic dynamics? (cosine similarity of the deltas) 
5. amplification. is speaker A or speaker B more salient? (norm of deltas divided by each other, norm_B/norm_A)
6. emotional inertia. how much change in emotional state is occuring within one speaker's responses. (cosine similarity between h[t] and h[t+1])

Annotations: just my own subjective opinions on each utterance to compare to

# Interpretation 
The results seem to capture some intuitive phenomena. >_<

# Notes and Considerations 

