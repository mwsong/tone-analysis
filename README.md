# Exploratory Analysis of Emotional Trajectories with Wav2Vec

# Overview 
This project analyzes how pretrained audio encoders represent speech variation during tense dialogue. Using selected scenes from movies (Weapons, Bugonia, and Blue Valentine), I tracked and attempted to mathematically quantify how, in conversations, peoples' tone and prosodic features evolve, respond to each other, and mirror each other. The goal was to examine whether these changes in embeddings could align with subjective perceptions. 

All results in weapons.ipynb, bugonia.ipynb, and blueval.ipynb. All functions used are in trajectory_analysis.py

# Ethical Concerns 
This project intentionally avoids real-world personal data and only uses fictional, public film dialogue. This analysis does not attempt to evaluate personal traits, worth, personality, and broader identity, just the acoustic patterns in a moment in time. People are incredibly complex and I don't intend any work to compete with intuition, rather to learn more about ourselves at best. Speech patterns are shaped by context, dialogue, narrative, performance, upbringing, and so so much more. 

# Motivation 
In real conversations, people continuously appraise and try to gauge the emotional space and interpersonal safety, sensing shifts in tension, tone, and alignment. My main questions and drivers towards this exploration were:
1. During appraisal, especially in moments of tension and stress, how do people form an idea of the emotional space?
2. How do we then externalize our intuition into a more formal system?
3. Could there be hidden, structured patterns in how we speak that we didn't have the tools to realize and learn from before, but now we can with transformers?

# Methods 
Audio embeddings: Wav2Vec2-mini 
Text embeddings(obsolete): miniLM (I ended up deciding that I wasn't sure how to concatenate the two embeddings together. If text embeddings were encoding on a different basis than the Wav2Vec, which captures prosody and tone, then I felt it could add noise to the information) 

Layers of Analysis:
1. mean value of raw embeddings 
2. self change, how much change is happening from one response to the next within one speaker. (delta = h[t] - h[t-1]) just subtracting the vectors
3. inter-speaker alignment. do the two speakers agree in the emotional space? (cosine similarity of h_A[t] and h_B[t] where h_B[t] would be the response to h_A[t])
4. speaker contagion. is the speakers change similar to the other speakers change? or is there some influence in the direction of their prosodic dynamics? (cosine similarity of the deltas) 
5. amplification. is speaker A or speaker B more salient? (norm of deltas divided by each other, norm_B/norm_A)
6. emotional inertia. how much change in emotional state is occurring within one speaker's responses. (cosine similarity between h[t] and h[t+1])

Annotations: just my own subjective opinions on each utterance to compare to
Spliced audio files with FL Studio

# Interpretation 
The results seem to capture some intuitive phenomena. The scene from Weapons was the most readable because the institutional and emotional stakes are explicit. The Weapons scene resulted in the most aligning results with my subjective interpretation. Other scenes did not align in any obvious ways at all, reinforcing the idea that more nuanced moods complicate quantitative interpretation. 
Again, there is no "failure" here, simply whether there's anything of interest. 

# Limitations 
1. Use of smaller encoders due to computational restraints
2. Manual audio splicing may introduce error
3. Single-scene analysis prevents proper baselines for interpreting the metrics
4. Film dialogue limits experimental control but avoids ethical concerns involving persons and personal data

# Notes and Considerations 
As machine learning systems increasingly model human behavior, it's important to balance exploration with humanity. There could be dimensions we couldn't quantify in our world that could be represented through transformers. This brings opportunity to learning more about ourselves and humanity rather than displacing our intuitive abilities. This project was an attempt to engage with this boundary. 
