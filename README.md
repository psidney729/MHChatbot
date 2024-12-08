# MHChatbot

[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
![python_version](https://img.shields.io/badge/Python-%3E=3.10-blue)

This is a mental health counseling chatbot using FastAPI, React, PostgreSQL, LangChain, LLM.

## Project structure

The project is composed of :

- a backend API server built with FastAPI located in the [backend](backend) dir.
- a frontend web app build with React and located in the [frontend](frontend) dir.


## My Solutions

### 1. LLM Chatbot (Mental Health Counseling Chatbot Using LLMs)
This project is a chatbot for mental health counseling powered by Large Language Models (LLMs). The system utilizes advanced AI techniques to provide users with empathetic, non-judgmental, and informative responses to mental health queries.

The project leverages the LangChain framework, a multi-agent architecture, and SQL semantic search to enable natural language interaction, data retrieval, and tailored guidance for users seeking mental health support.


#### Key Features
- LangChain Integration: Utilizes LangChain to manage and chain interactions with LLMs.
- Multi-Agent Collaboration: Implements multiple agents, each with specific tasks, to enhance modularity and performance.
- SQL Semantic Search: Enables semantic understanding of natural language queries to retrieve information from an SQL database.
- Natural Language Querying: Users can input queries in plain language, which are then translated into SQL by the system.

![Mental Health Counseling LLM Chatbot schema](3.%20chatbotllm.png)

⚠️ Important : Just domain-specific knowledge base for mental health counseling.(conversationDB.csv)

### 2. Semantic Search
Semantic search is a technique that leverages embeddings to match natural language queries to relevant information based on meaning rather than exact keyword matches. In this project, semantic search enables the chatbot to retrieve mental health resources (e.g., articles, coping strategies, or FAQs) that best align with a user's query.

#### Key Features
- Embedding Representation : OpenAI's embedding model (e.g., text-embedding-ada-002) is used to convert both the natural language query and the data into high-dimensional vector representations. These embeddings capture the semantic meaning of the text.

- Vector Storage : The embeddings for the resource database are precomputed and stored in a vector database, such as FAISS (Facebook AI Similarity Search) or another efficient storage system. Each resource (e.g., an article, technique, or FAQ) is represented as a vector.

- Query Encoding : When a user submits a query, the chatbot converts the input into an embedding using the same OpenAI embedding model.

- Similarity Search : The system calculates the similarity (e.g., cosine similarity) between the query embedding and all stored embeddings in the vector database. The closest matches are returned as the most relevant resources.

- Result Retrieval : The chatbot retrieves the top-ranked resources and incorporates them into its response to the user.

![Semantic Search](1.%20semanticsearch.png)

⚠️ Important : Just semantic search for mental health counseling - conversationDB.csv

### 3. Classification(Regresstion) Using Machine Learning Model
The primary objective of this task is to build a machine learning (ML) model capable of predicting or inferring a specific piece of information using a given dataset. This involves selecting a numeric or categorical target variable to predict, identifying appropriate input features, and implementing the necessary steps to train and evaluate the ML model.

#### Example prediction
- Likelihood Prediction: Estimating the probability of a patient presenting with a particular problem based on their demographic or medical data.
- Response Type Prediction: Determining the expected type of response from a healthcare provider (e.g., direct advice vs. general information) given a particular patient issue.

#### Current State
Due to limitations with the current database infrastructure, my implementation focuses on frontend and API integration rather than directly building and training the ML model at this stage. If there is advanced dataset to analyze, I will implement the ML model.

![Classification(Regresstion) Using Machine Learning Model](2.%20mlclassification.png)

⚠️ Important : Database is not ready yet.