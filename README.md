# Adapting Large-Scale Speaker-Independent Automatic Speech Recognition to Dysarthric Speech

### Description

Code accompanying masters minor dissertation which investigates different approaches to improving dysarthric speech recognition in a Deep Learning context. More specifically, [Deep Speech](https://github.com/mozilla/DeepSpeech) v0.9.3 is fine-tuned to the [UASpeech](http://www.isle.illinois.edu/sst/data/UASpeech/) dataset. In addition to fine-tuning, layer freezing, data augmentation, and re-initialization are investigated. 

### Training and fine-tuning using Mozilla Deep Speech

To train and fine-tune models using Mozilla Deep Speech, the procedure outlined in the "steps" folder was adhered to. This was based on the Deep Speech [documentation](https://deepspeech.readthedocs.io/en/r0.9/TRAINING.html). An AWS P2 instance running a Base Deep Learning AMI and harnessing a Tesla K80 GPU was used. The *splits.py* scripts in the "deep-speech" folder are responsible for splitting the data and writing to csv. 

### Running inference on the fine-tuned models

The [source code](https://deepspeech.readthedocs.io/en/r0.9/Python-Examples.html#py-api-example) for running inference using Deep Speech was used as a starting point. It was modified to calculate the word error rate (WER) and character error rate (CER) for all examples from the csv files created by *splits.py*. It also stratified results by intelligibility level and speaker. This was coded up separately for the control and dysarthric data. 

### Freezing layers

Layer freezing is not currently an option in Mozilla Deep Speech. To achieve it, the [approach](https://github.com/onnoeberhard/deepspeech-transfer/tree/transfer-2) used by Eberhard and Zesch (2021) was applied. This primarily consisted of setting "trainable=False" for the desired layers. 

### Encoder-decoder model

The encoder-decoder model is coded from scratch, but inspired by an [article](https://keras.io/examples/nlp/lstm_seq2seq/) by Francois Chollet. The crux of the model is in *main.py*. Feature extraction using MFCCs can be found in *data.py*. The WERs and CERs are calculated using *levenshtein.py* and *results.py*.

### Figures

Exploratory data analysis of the UASpeech dataset as well as many of the plots used in the thesis are provided. 

### References

* Hannun, Awni, Carl Case, et al. (2014). Deep Speech: Scaling up end-to-end speech recognition. arXiv: 1412.5567 [cs.CL].
* Mozilla (2020). Training Your Own Model. URL: https://deepspeech.readthedocs.io/en/r0.9/TRAINING.html (visited on 10/28/2021).
* Kim, Heejin et al. (2008). “Dysarthric speech database for universal access research”. English (US). In: Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, pp. 1741–1744. issn: 2308-457X.
* Chollet, Francois (2017). Character-level recurrent sequence-to-sequence model. URL: https://keras.io/examples/nlp/lstm_seq2seq/ (visited on 09/16/2021).
* Eberhard, Onno and Torsten Zesch (Sept. 2021). “Effects of Layer Freezing on Transferring a Speech Recognition System to Under-resourced Languages”. In: Proceedings of the 17th Con- ference on Natural Language Processing (KONVENS 2021). Du ̈sseldorf, Germany: KONVENS 2021 Organizers, pp. 208–212. url: https://aclanthology.org/2021.konvens-1.19.
