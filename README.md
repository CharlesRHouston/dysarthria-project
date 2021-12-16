# Adapting Large-Scale Speaker-Independent Automatic Speech Recognition to Dysarthric Speech

### Description

Code accompanying masters minor dissertation which investigates different approaches to improving dysarthric speech recognition. More specifically, Deep Speech is fine-tuned to the UASpeech dataset. In addition to fine-tuning, layer freezing, data augmentation, and re-initialization are investigated. 

### Training and fine-tuning using Mozilla Deep Speech

To train and fine-tune models using Mozilla Deep Speech, the procedure outlined in "steps" folder was adhered to. An AWS P2 instance running a Base Deep Learning AMI was used. The 

### Running inference on the fine-tuned models

The 

### Freezing layers

Layer freezing is not currently an option in Mozilla Deep Speech. To achieve it, the [approach](https://github.com/onnoeberhard/deepspeech-transfer) outlined by Eberhard and Zesch (2021) was applied. This primarily consisted of setting "trainable=False" for the desired layers. 

### Encoder-decoder model

### Folders & Scripts


### References

* Hannun, Awni, Carl Case, et al. (2014). Deep Speech: Scaling up end-to-end speech recognition. arXiv: 1412.5567 [cs.CL].
* Mozilla (2020). Training Your Own Model. URL: https://deepspeech.readthedocs.io/en/r0.9/TRAINING.html (visited on 10/28/2021).
* Kim, Heejin et al. (2008). “Dysarthric speech database for universal access research”. English (US). In: Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, pp. 1741–1744. issn: 2308-457X.
* Chollet, Francois (2017). Character-level recurrent sequence-to-sequence model. URL: https://keras.io/examples/nlp/lstm_seq2seq/ (visited on 09/16/2021).
* Eberhard, Onno and Torsten Zesch (Sept. 2021). “Effects of Layer Freezing on Transferring a Speech Recognition System to Under-resourced Languages”. In: Proceedings of the 17th Con- ference on Natural Language Processing (KONVENS 2021). Du ̈sseldorf, Germany: KONVENS 2021 Organizers, pp. 208–212. url: https://aclanthology.org/2021.konvens-1.19.
