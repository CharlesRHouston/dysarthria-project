## Adapting Automatic Speech Recognition to Dysarthric Speech

#### Description

Code accompanying masters minor dissertation which investigates different approaches to improving dysarthric speech recognition. More specifically, Deep Speech is fine-tuned to the UASpeech dataset. In addition to fine-tuning, layer freezing, data augmentation and re-initialization are investigated. 

#### Folders & Scripts

* *deep-speech:* 
  * split.py: split the data into train/validation/test sets for both dysarthric and control data 
  * inference.py: taken from Deep Speech source code and modified to run evaluation on the UASpeech dataset
* *encoder-decoder:*  
  * main.py: crux of the encoder-decoder model
  * levenshtein.py: levenshtein distance used in results.py
  * data.py: feature extraction
  * results.py: speaker and intelligibility based WERs & CERs
* *exploratory-data-analysis:* 
  *  UASpeech.py: intelligibility, duration, and raw waveform plots of the UASpeech data
* *freeze:* 
  * DeepSpeech.py: taken from Deep Speech source code and modified to achieve layer freezing by setting trainable=False for the first three layers
  * train.py: taken from Deep Speech source code and modified to import DeepSpeech.py
* *plots:* 
  * creation of plots used in the dissertation

#### References

* Deep Speech
* Encoder-decoder
* UASpeech
