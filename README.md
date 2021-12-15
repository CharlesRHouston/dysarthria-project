# Adapting Large-Scale Speaker-Independent Automatic Speech Recognition to Dysarthric Speech

### Description

Code accompanying masters minor dissertation which investigates different approaches to improving dysarthric speech recognition. More specifically, Deep Speech is fine-tuned to the UASpeech dataset. In addition to fine-tuning, layer freezing, data augmentation and re-initialization are investigated. 

### Folders & Scripts

* deep-speech: 
  * *split.py:* split the data into train/validation/test sets for both dysarthric and control data and write to csv
  * *inference.py:* taken from Deep Speech source code and modified to run evaluation on the UASpeech dataset (WER and CER at speaker and intelligibility levels)
* encoder-decoder:  
  * *main.py:* crux of the encoder-decoder model implemented in Tensorflow
  * *levenshtein.py:* levenshtein distance used in results.py
  * *results.py:* speaker and intelligibility based WERs & CERs
  * *data.py:* feature extraction using MFCCs
* exploratory-data-analysis: 
  *  *UASpeech.py:* intelligibility level, duration distribution, and raw waveform plots of the UASpeech data
* freeze:
  * *DeepSpeech.py:* taken from Deep Speech source code and modified to achieve layer freezing by setting trainable=False for the first three layers
  * *train.py:* taken from Deep Speech source code and modified to import DeepSpeech.py
* plots: 
  * creation of all plots used in the dissertation
* steps:
  * setting up an AWS instance with the appropriate modules and versions
  * training and exporting pb models using Deep Speech
  * inference using the modified code

### References

* Hannun, Awni, Carl Case, et al. (2014). Deep Speech: Scaling up end-to-end speech recognition. arXiv: 1412.5567 [cs.CL].
* Mozilla (2020). Training Your Own Model. URL: https://deepspeech.readthedocs.io/en/r0.9/TRAINING.html (visited on 10/28/2021).
* Kim, Heejin et al. (2008). “Dysarthric speech database for universal access research”. English (US). In: Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, pp. 1741–1744. issn: 2308-457X.
* Chollet, Francois (2017). Character-level recurrent sequence-to-sequence model. URL: https://keras.io/examples/nlp/lstm_seq2seq/ (visited on 09/16/2021).
* Eberhard, Onno and Torsten Zesch (Sept. 2021). “Effects of Layer Freezing on Transferring a Speech Recognition System to Under-resourced Languages”. In: Proceedings of the 17th Con- ference on Natural Language Processing (KONVENS 2021). Du ̈sseldorf, Germany: KONVENS 2021 Organizers, pp. 208–212. url: https://aclanthology.org/2021.konvens-1.19.
