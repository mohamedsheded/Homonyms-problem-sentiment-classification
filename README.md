# Sentiment-Analysis-for-Homonyms-Problem
**Homonyms** are words that **share the same spelling** or form (characters) but possess **distinct meanings**. For instance, the term **"bank"** can assume two disparate contexts, denoting both a **financial institution** and **the edge of a river**.

These homonyms hold **significant relevance** in **sentiment analysis**, given their capacity to **alter a sentence's meaning** or **emotional tone** entirely. Consider the following examples that highlight this challenge:

| Sentence | Label |
| --- | --- |
| I **hate** the selfishness in you | NEGATIVE |
| I **hate** anyone who hurts you | POSITIVE |

In the first sentence, the word **"hate"** renders the sentiment as **NEGATIVE**. Conversely, the same word, "hate" appears in the second sentence, shaping the sentiment of the sentence as **POSITIVE**. This poses a **considerable challenge** to models relying on **fixed word embeddings**. Therefore, **employing contextualized embeddings leveraging attention mechanisms from transformers becomes crucial to grasping the comprehensive context within a sentence**.

# Tools Used
The project is implemented using the following Python packages:

| Package | Description |
| --- | --- |
| NumPy | Numerical computing library |
| Pandas | Data manipulation library |
| Matplotlib | Data visualization library |
| Sklearn | Machine learning library |
| pytorch | Open-source machine learning framework |
| Transformers | Hugging Face package contains state-of-the-art Natural Language Processing models |
| Datasets | Hugging Face package contains datasets |

# Dataset
The [Stanford Sentiment Treebank (SST)](https://huggingface.co/datasets/sst2) is a corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language. The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of **11,855** single sentences extracted from **movie reviews**. It was parsed with the Stanford parser and includes a total of **215,154** unique phrases from those parse trees, each annotated by 3 human judges.

Binary classification experiments on full sentences (negative or somewhat negative vs somewhat positive or positive with neutral sentences discarded) refer to the dataset as SST-2 or SST binary.

The dataset contains **3** features [`idx`, `sentence`, and `label`] and it comes in 3 different splits [train, validation, and test]. Here is the number of samples per split:


| Split | Number of Samples |
| --- | --- |
| train | 67349 |
| validation | 872 |
| test | 1821 |

# Methodology
## Dataset Preparation
1. Data generation using ChatGPT for making ~ 50 examples on data-like given on the project examples 
2. plotting some graphs seeing class inbalancing on (positives/negatives) counts
3. feature engineering: `words_per_sentences` to see the avg len of each input to our model:
    3.1 removing senetence larger than 30 words (~2000) from 70,000
4. Label2id and id2label for the easy usage "added to model config"
5. the final Train/Test Spilt is

| Split | Number of Samples |
| --- | --- |
| train | 58851 |
| validation | 2180 |
| test | 4359 |


This phase encompasses dataset preparation preceding the modeling phase. It involves transforming the `Sentence` column into numerical representations utilizing the `Tokenizer` class from the `transformers` library. Subsequently, the integer sequences are padded to confirm to the maximum sequence length within the dataset.

## Models Selection
1. The two models selected are **Bert-base-uncased** and **distilbert-base-uncased**. The rationale behind choosing these models lies in the necessity to use **contextualized embeddings** instead of **fixed-embedding**  
2. **distilbert-base-uncased** : is the distilled(Teacher/Student) version of **BERT**. DistilBERT offers a 40% reduction in memory requirements **(67 million parameters)** compared to the standard BERT base (110 million parameters) and almost no drawbacks in effiency 
3. The core reason for using DistilBERT is when testing with `transformer_pipeline` without choosing any model; i Found a finetuned version of DistilBERT **distilbert/distilbert-base-uncased-finetuned-sst-2-english**


# Results
1. Training Args and models 
## Comparison of Training Arguments: BERT vs. DistilBERT

| **Model**      | **Number of Epochs** | **Learning Rate** | **Batch Size** | **Max Sequence Length** | **Notes**                                                                                  |
|-----------------|----------------------|--------------------|----------------|--------------------------|--------------------------------------------------------------------------------------------|
| **BERT**       | 2                    | 2e-5               | 64             | 512                      | Standard BERT configuration for fine-tuning tasks.                                         |
| **DistilBERT** | 3                    | 1e-5               | 32             | 128                      | Training arguments align with the model card of `distilbert-base-uncased-finetuned-sst-2-english`. |

### Notes:
- **Number of Epochs**: DistilBERT uses more epochs (3 vs. 2) to compensate for its smaller model size compared to BERT.
- **Learning Rate**: A smaller learning rate (1e-5) is used for DistilBERT compared to BERT (2e-5) for finer adjustments during training.
- **Batch Size**: BERT uses a larger batch size (64 vs. 32), likely due to its larger memory requirements.

## Comparison of Training Results: DistilBERT vs. BERT

| **Model**      | **Epoch** | **Training Loss** | **Validation Loss** | **Accuracy** | **F1 Score** |
|-----------------|-----------|-------------------|----------------------|--------------|--------------|
| **DistilBERT** | 1         | 0.223900          | 0.173022             | 0.936239     | 0.936262     |
|                 | 2         | 0.146800          | 0.141964             | 0.952294     | 0.952328     |
|                 | 3         | 0.115600          | 0.147620             | 0.952752     | 0.952784     |
| **BERT**       | 1         | 0.252000          | 0.151614             | 0.940609     | 0.940547     |
|                 | 2         | 0.153700          | 0.152472             | 0.948181     | 0.948215     |

### Observations:
- **DistilBERT** achieved higher accuracy and F1 scores by the end of training (epoch 3).
- **DistilBERT’s** training process had lower training loss in later epochs, likely due to the smaller model size and additional epoch allowing better fine-tuning.



## Classification Report:
| Metric          | DistilBERT (Left) | BERT (Right) |
|------------------|-------------------|--------------|
| **Class 0**     |                   |              |
| Precision        | 0.93              | 0.94         |
| Recall           | 0.94              | 0.94         |
| F1-Score         | 0.93              | 0.94         |
| Support          | 1925              | 5956         |
| **Class 1**     |                   |              |
| Precision        | 0.95              | 0.95         |
| Recall           | 0.94              | 0.95         |
| F1-Score         | 0.95              | 0.95         |
| Support          | 2434              | 7514         |
| **Overall**     |                   |              |
| Accuracy         | 0.94              | 0.95         |
| Macro Avg        | 0.94              | 0.95         |
| Weighted Avg     | 0.94              | 0.95         |
| Total Support    | 4359              | 13470        |


## Models Testing
| Text                                       | Label     | BERT Prediction | DistilBERT Prediction |
|--------------------------------------------|-----------|-----------------|------------------------|
| I love you                                 | POSITIVE  |    POSITIVE        |     POSITIVE                   |
| I hate you                                 | NEGATIVE  | NEGATIVE  |   NEGATIVE   |
| I hate the selfishness in you              | NEGATIVE  | NEGATIVE                |    NEGATIVE                    |
| I hate anyone hurts you                    | POSITIVE  |   NEGATIVE              |    POSITIVE                    |
| I hate anyone hurting you                  | POSITIVE  |       NEGATIVE          |    POSITIVE                    |

`DistilBERT` is able to correctly predict the homonyms examples while BERT cannot this duo to some reasons :
1. more synthetic Data are added during training to DistilBERT
2. DistilBERT is trained with lower Learning rate + 1 Additional Epoch
3. DistilBERT has 40% reduction in size so it has lower chances to overfit with respect to BERT
   
# Conclusion
1. the Decision for going to  fixed embedding models(LSTM) won't work well on this problem duo to the Fixed embeddings
2. Finetuning BERT for the same `training args` and adding the `new generated data~50 example` used in DitilBERT will probably go to the same results 
3. Fixed embeddings might suffice when dealing with `straightforward` data, offering good performance while requiring less memory than transformer-based models. However, in cases where the data is more complex, as demonstrated in our test cases, leveraging a transformer-based model with a self-attention mechanism can yield performance improvements. It's crucial to note that this advantage comes at the expense of a higher memory footprint.

# Setfit transformer architecture
![image](https://github.com/user-attachments/assets/c12746ca-98fb-4c57-a792-2b69a3112cf4)
![image](https://github.com/user-attachments/assets/561b3f1c-d673-4d5e-8b5a-1d1af3b534bd)


# References
- [DistilBERT SST-2 Fine-tuned Model](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) - Pre-trained DistilBERT fine-tuned for sentiment analysis.
- [Hugging Face Transformers - Fine-tuning Models](https://huggingface.co/transformers/v4.21.1/training.html) - Official documentation for training and fine-tuning transformers.
- [Hugging Face Training Guide](https://huggingface.co/docs/transformers/training) - Step-by-step training and fine-tuning examples.
- [Pre-trained BERT Models](https://huggingface.co/models?filter=bert) - List of BERT models available for fine-tuning.
- [Hugging Face Text Classification Example](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb) - Colab example for text classification using BERT.

