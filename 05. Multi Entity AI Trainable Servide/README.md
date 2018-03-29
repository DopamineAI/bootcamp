# Multi Entity AI Trainable Service

In this sample consumer A wishes to have an image classifier for digits, 
while assuming there are three relevant AI modules out there in the ecosystem that were used in other projects:
- A Retina module - specializing in efficiently extracting low-level features from images, owned by Entity “R”.  
- A Visual Cortex module - specializing in efficiently extracting high level features from low level modules, owned by Entity “V”. 
- A Classifier module - specializing in providing good estimations about which class does each sample belong to, owned by Entity “C”.

The consumer codes that she expects the data to flow in a pipeline created by these 3 entities, and state the reward she is willing to pay for that pipeline. Under the hood, the Dopamine network takes care of:
- Reward negotiations
- Pipeline creation
- Matching validations
- Efficient data flow
- Reward book-keeping
- And more…

<img src='https://github.com/DopamineAI/bootcamp/blob/master/img/05_all.png'>


# Code Samples

- [Entity A (Client)](https://github.com/DopamineAI/bootcamp/blob/master/05.%20Multi%20Entity%20AI%20Trainable%20Servide/client.ipynb)
- [Entity A (Retina)](https://github.com/DopamineAI/bootcamp/blob/master/05.%20Multi%20Entity%20AI%20Trainable%20Servide/retina.ipynb)
- [Entity V (VisualCortex)](https://github.com/DopamineAI/bootcamp/blob/master/05.%20Multi%20Entity%20AI%20Trainable%20Servide/visualcortex.ipynb)
- [Entity C (classifier)](https://github.com/DopamineAI/bootcamp/blob/master/05.%20Multi%20Entity%20AI%20Trainable%20Servide/classifier.ipynb)
