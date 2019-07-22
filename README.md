# ON-LSTM for Machine Translation

Literature Review
=====

// 1. Describe the architecture and the difference between the normal LSTM and the Ordered Neuron

// 2. Motivate the advantages and why you proposed that it'll work for machine translation

// 3. Neural machine translation are encoder-decoder architectures, describe it. The state-of-the-art is Transformer based architecture but the previous state-of-the-art is the stacke LSTM-based architecture, describe the LSTM-based architecture. 

// 4. (Our) Proposal is that the tree induced in the ON-LSTM `yada yada yada`, so it will perform better when transferring knowledge from the source language with `yada yada yada` to enable more `yada yada yada`-driven translations in the target language


Proposed Approach
=====

/* 

A bit of a copy of point (4) from "literature review" but with more details, e.g. 

 - what is the cumsum doing to the master gates
 - how are the trees extracted (remember,  Stanley described how the extract tree something .py works)
 - how are the hidden states (affected by the master gates) gets transfered from the encoder-decoder network, currently, just transferred like LSTM (but please rephrase this to make it sound better)
 
 - How to show that the proposed architecture works better? For which task? 
 - Be explicit in the "hypotheses" that you want to test with the proposed architecture and the task. 

*/

Experimental Setup
====

- Describe dataset, why it fits to prove/show the strength/weakness of the architecture.
  - Describe what preprocessing is done to the dataset
  
- Describe the tools used in the experiment, torch version, opennmt version, 
  - If there's any other changes made, describe them e.g. script or something.
  
- Describe the model built using data and tools listed above, i.e. architecture type, hyperparameters, validation routine, early stopping criteria, evaluation metric 
  
Results
====

- Explain what results are achieved based on the evaluation metric, quantitatively 
- Find possible reason why it's doing better/worse
- Look into the output of the models to spot any errors that may appear qualitatively

Summary
====

 - Summarize

Future Work
====

  - Describe what needs to be done to improve "Results" or prove/disprove the hypotheses if the project is on-going. 
 
 
