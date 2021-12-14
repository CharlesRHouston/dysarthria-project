## Adapting Automatic Speech Recognition to Dysarthric Speech

#### Description

Code accompanying my masters minor dissertation investigates different approaches to improving dysarthric speech recognition. More specifically, Deep Speech is fine-tuned to the UASpeech dataset. In addition to fine-tuning, layer freezing, data augmentation and re-initialization are investigated. 

#### Folders

* *deep-speech:* 
  * split.py: split the data into train/validation/test sets for both dysarthric and control data 
  * inference.py: taken from Deep Speech source code and modified to run evaluation on the UASpeech dataset
* *encoder-decoder:*  
  * code to build and evaluate an encoder-decoder recurrent neural network
* *exploratory-data-analysis:* different plots exploring the UASpeech dataset
* *freeze:* modifications made to certain Deep Speech files to achieve layer freezing
* *plots:* creation of dissertation plots

#### References
