# Sentiment Classification using Transfer learning 

## Motivation 

I wanted to - 
1. Use huggingface libraries and pre-trained models 
2. Use transfer learning for Text database. 
3. Learn the word embeddings

## Objective 

We have a movie review dataset form kaggle (small dataset). It is a labelled data and we want to be able to predict the labels.

## Data Description 

This is about the films that were released at about the asame time . 

We have been provided the reviews and labels as to whether the sentiment is positive or negative. 

nearly 5500 sentences. 

## why transfer learning ? 

The dataset is so small in size that training a full ML model will lead to overfitting so early that we will never be able to generalize anything about the data. It is therefore necessary to use learned features. 

### What bert model brings ? 

1. It brings information of text embeddings. 
2. It brings a knowledge about the world and how they correlate to the other words. 

### what does huggingface brings? 

1. It brings the architecture and ease of training using predefined and documented pipelines. 

## Structure. 

1. data csv is in '/data/sentiment_csv.csv'.
2. Python notebook to do the data analysis. 

## Decisions made. 

I will write any design decisions I made here. 

## Visuals to illustrate the understanding

## Final Results. 


