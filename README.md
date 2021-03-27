# Text_Classification_with_fine_tuned_BERT_model

This is a simple code customized on a pretained Transformer model using:
- Google's Tensorflow Pretrained Bi-directional Transformer model with large text corpus, including Wikipedia
- English
- Tensorflow 2
- Keras model buildout

Samples are from Kaggle Quora dataset 2018, of which 10,000 were selected as train set, 1,000 as val set.
- After training data is slightly overfit, but still able to pass test
- Hyperparameters used: Adam, learning rate 2e-5 (small), train (mini)-batch size 32

Actual output 'preds', 2021.03.27 ran on Google Colab
- array([[5.6299049e-04],
- [3.3487119e-02],
- [4.9054455e-03],
- [4.9818565e-05],
- [2.1208545e-04],
- [3.5636741e-01]], dtype=float32)
     
Results summary
- #1 "Do you like flowers?" 5.629e-04, Sincere
- #2 "Why all the bad guys wear a black hat?" 3.348e-02, Toxic
- #3 "Why so many taxi drivers are so rude?" 4.905e-03, Sincere
- #4 "May I have your email ID?" 4.981e-05, Sincere
- #5 "How are you today?" 2.120e-04, Sincere
- #6 "Why are you such a bad ass?" 3.563e-01, Toxic

FINAL RESULT
- All passed!! Because all 6 sentences are clasified correctly

# Justification

After trained for only 15 minutes on a Google Colab P100 GPU, the model is slightly overfit, but with a already well-trained BERT model like this one, it should not work too badly, so tested briefly with 6 Quora-like questions:

sample_example = ["Do you like flowers?",\
                  "Why all the bad guys wear a black hat?",\
                  "Why so many taxi drivers are so rude?",\
                  "May I have your email ID?",\
                  "How are you today?",\
                  "Why are you such a bad ass?"]
                  
And the AI can do a perfect job to classify between "Sincere" and "Toxic". In particular, the hardest question is the third question "Why so many taxi drivers are so rude?". As this question looks toxic on the first glance, for instance, navie word based classification will easily misclassify to toxic between the word "rude" contribute to a strong negative. However, BERT is already well trained, and having good knowledge, about the "meaning" of any sentence I feed to it, so it somehow see the overall positive meaning (if not neural) of the sentence and decided it is "Sincere", which is obviously correct from human standpoint.

