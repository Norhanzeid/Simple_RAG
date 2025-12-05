from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

# NLP content - Extended for 20+ pages
nlp_content = """
Natural Language Processing (NLP) - Complete Guide

Chapter 1: Introduction to Natural Language Processing

Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human languages in a valuable way.

1.1 What is Natural Language Processing?

NLP combines computational linguistics—rule-based modeling of human language—with statistical, machine learning, and deep learning models. These technologies enable computers to process human language in the form of text or voice data and to 'understand' its full meaning, complete with the speaker or writer's intent and sentiment.

The field of NLP has evolved significantly over the decades. Early approaches relied heavily on hand-crafted rules and linguistic knowledge. Modern NLP leverages vast amounts of data and powerful machine learning algorithms to automatically learn patterns and representations from text.

1.2 Historical Development of NLP

The history of NLP can be traced back to the 1950s when Alan Turing published his famous paper proposing the Turing Test as a criterion of intelligence. Since then, NLP has gone through several paradigm shifts:

- 1950s-1960s: Rule-based systems and symbolic approaches
- 1970s-1980s: Introduction of statistical methods
- 1990s-2000s: Machine learning approaches become dominant
- 2010s: Deep learning revolution transforms the field
- 2020s: Large language models and transformer architectures

1.3 Core Applications of NLP

Machine Translation: Converting text from one language to another (e.g., Google Translate, DeepL). Modern neural machine translation systems can produce remarkably fluent translations across hundreds of language pairs.

Sentiment Analysis: Determining the emotional tone behind words. This is widely used in social media monitoring, customer feedback analysis, and market research.

Chatbots and Virtual Assistants: Conversational AI systems like Siri, Alexa, Google Assistant, and ChatGPT that can understand and respond to natural language queries.

Text Summarization: Automatically creating summaries of large documents. This includes both extractive summarization (selecting key sentences) and abstractive summarization (generating new text).

Named Entity Recognition: Identifying and categorizing key information in text such as names, organizations, locations, dates, and quantities.

Question Answering Systems: Systems that can answer questions posed in natural language by retrieving or generating appropriate responses.

Information Extraction: Automatically extracting structured information from unstructured text sources.

Speech Recognition: Converting spoken language into text, enabling voice interfaces and transcription services.

Chapter 2: Fundamental NLP Tasks and Techniques

2.1 Tokenization

Tokenization is the process of breaking down text into smaller units called tokens. These tokens can be words, characters, or subwords. It is a fundamental step in NLP as it converts raw text into a format that can be processed by machine learning algorithms.

Word Tokenization Example:
Input: "Natural Language Processing is amazing!"
Output: ["Natural", "Language", "Processing", "is", "amazing", "!"]

Subword Tokenization:
Modern approaches like Byte Pair Encoding (BPE) and WordPiece break words into subword units, which helps handle rare words and morphological variations. For example, "unhappiness" might be tokenized as ["un", "happiness"].

2.2 Part-of-Speech Tagging

Part-of-speech (POS) tagging is the process of marking up words in a text as corresponding to a particular part of speech, such as noun, verb, adjective, etc., based on both its definition and context.

Example:
"The quick brown fox jumps over the lazy dog"
The/DET quick/ADJ brown/ADJ fox/NOUN jumps/VERB over/PREP the/DET lazy/ADJ dog/NOUN

POS tagging is crucial for many downstream NLP tasks as it provides grammatical information about each word.

2.3 Named Entity Recognition (NER)

NER is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as:
- PERSON: Individual names
- ORGANIZATION: Companies, institutions
- LOCATION: Cities, countries, geographical features
- DATE: Temporal expressions
- MONEY: Monetary values
- PERCENT: Percentages
- PRODUCT: Product names

Example:
"Apple Inc. announced that Tim Cook will visit Paris in December 2024."
Entities: [Apple Inc.: ORG, Tim Cook: PERSON, Paris: LOC, December 2024: DATE]

2.4 Dependency Parsing

Dependency parsing analyzes the grammatical structure of a sentence and establishes relationships between words. It creates a tree structure where each word depends on another word (except the root).

For example, in "The cat sat on the mat":
- "sat" is the root
- "cat" depends on "sat" (subject)
- "The" depends on "cat" (determiner)
- "on" depends on "sat" (prepositional modifier)
- "mat" depends on "on" (object of preposition)
- "the" depends on "mat" (determiner)

2.5 Coreference Resolution

Coreference resolution identifies when different expressions in a text refer to the same entity. For example:
"John went to the store. He bought milk."
The system must recognize that "He" refers to "John".

Chapter 3: Text Preprocessing and Normalization

3.1 Lowercasing

Converting all characters in the text to lowercase ensures uniformity. This helps in reducing the vocabulary size and treating words like "Apple" and "apple" as the same token. However, lowercasing should be applied judiciously as it can remove important information (e.g., distinguishing between "Apple" the company and "apple" the fruit).

3.2 Removing Punctuation and Special Characters

Eliminating punctuation marks and special characters that may not contribute to the meaning of the text in certain applications. However, punctuation can carry semantic information (e.g., question marks, exclamation points) and should be retained for tasks like sentiment analysis.

3.3 Stop Words Removal

Stop words are common words like "the", "is", "in", "and", "of", "to", etc., that typically don't carry significant meaning by themselves. Removing them can help:
- Reduce the dimensionality of the data
- Speed up processing
- Focus on more meaningful words

However, stop words can be important for certain tasks (e.g., "to be or not to be" loses all meaning without stop words).

3.4 Stemming and Lemmatization

Stemming: Reduces words to their root form by removing suffixes using heuristic rules. Examples:
- "running" → "run"
- "flies" → "fli"
- "studies" → "studi"

Popular stemmers include Porter Stemmer and Snowball Stemmer.

Lemmatization: Converts words to their base or dictionary form (lemma) considering the morphological analysis and context. Examples:
- "running" → "run"
- "better" → "good"
- "was" → "be"

Lemmatization is generally more accurate than stemming but computationally more expensive.

3.5 Spell Correction

Automatically correcting spelling errors in text using techniques like:
- Edit distance (Levenshtein distance)
- Probabilistic models
- Context-aware correction using language models

3.6 Text Normalization

Normalizing text variations such as:
- Expanding contractions ("don't" → "do not")
- Converting numbers to words or vice versa
- Standardizing date and time formats
- Handling emojis and emoticons

Chapter 4: Text Representation and Feature Engineering

4.1 Bag of Words (BoW)

Bag of Words is a simple representation where text is represented as a multiset of its words, disregarding grammar and word order but keeping multiplicity.

Example:
Document 1: "I love NLP"
Document 2: "NLP is amazing"

Vocabulary: ["I", "love", "NLP", "is", "amazing"]

BoW representation:
Doc 1: [1, 1, 1, 0, 0]
Doc 2: [0, 0, 1, 1, 1]

Limitations:
- Loses word order information
- Ignores semantics
- High dimensionality for large vocabularies

4.2 TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection of documents.

Term Frequency (TF): How often a term appears in a document
TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)

Inverse Document Frequency (IDF): How rare or common a term is across all documents
IDF(t) = log(Total number of documents / Number of documents containing term t)

TF-IDF(t, d) = TF(t, d) × IDF(t)

TF-IDF helps identify words that are important to a document but not common across all documents.

4.3 N-grams

N-grams are contiguous sequences of n items from a text. They capture local word order information:
- Unigrams (1-gram): Individual words ["Natural", "Language", "Processing"]
- Bigrams (2-gram): Word pairs ["Natural Language", "Language Processing"]
- Trigrams (3-gram): Three-word sequences ["Natural Language Processing"]

N-grams are useful for capturing phrases and local context.

4.4 Word Embeddings

Word embeddings are dense vector representations of words in a continuous vector space where semantically similar words are mapped to nearby points.

Word2Vec: Developed by Google, uses neural networks to learn word associations through two architectures:
- CBOW (Continuous Bag of Words): Predicts a word given its context
- Skip-gram: Predicts context words given a target word

GloVe (Global Vectors): Developed by Stanford, focuses on word co-occurrence statistics from the entire corpus. It combines the benefits of matrix factorization and local context window methods.

FastText: Extension of Word2Vec developed by Facebook that represents each word as a bag of character n-grams. This allows it to generate embeddings for out-of-vocabulary words and better handle morphologically rich languages.

Properties of Word Embeddings:
- Semantic similarity: Similar words have similar vectors
- Analogical relationships: vec(king) - vec(man) + vec(woman) ≈ vec(queen)
- Dimensionality reduction: Typically 100-300 dimensions instead of vocabulary size

4.5 Contextualized Word Embeddings

Traditional word embeddings assign the same vector to a word regardless of context. Contextualized embeddings generate different vectors based on the surrounding words:

ELMo (Embeddings from Language Models): Uses bidirectional LSTMs to generate context-dependent representations.

BERT embeddings: Extract contextualized representations from BERT's transformer layers, capturing bidirectional context.

Chapter 5: Classical Machine Learning for NLP

5.1 Naive Bayes Classifier

Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption of independence between features. Despite this unrealistic assumption, it works surprisingly well for text classification tasks like spam detection and sentiment analysis.

The classifier calculates P(class|document) for each class and selects the class with the highest probability.

Advantages:
- Fast training and prediction
- Works well with small datasets
- Simple to implement

Disadvantages:
- Independence assumption rarely holds
- Can be outperformed by more sophisticated models

5.2 Support Vector Machines (SVM)

SVM finds the optimal hyperplane that maximizes the margin between different classes. For NLP tasks, text is typically represented using TF-IDF or other vectorization methods.

SVMs work well for:
- Text categorization
- Named entity recognition
- Sentiment analysis

They are particularly effective with high-dimensional data and when classes are well-separated.

5.3 Decision Trees and Random Forests

Decision trees learn a tree structure where each node represents a feature test and each leaf represents a class label. Random forests combine multiple decision trees to improve accuracy and reduce overfitting.

Applications in NLP:
- Text classification
- Feature importance analysis
- Handling non-linear relationships

5.4 Hidden Markov Models (HMM)

HMMs are statistical models that assume the system being modeled is a Markov process with hidden states. They have been widely used for:
- Part-of-speech tagging
- Speech recognition
- Named entity recognition

An HMM consists of:
- States (often hidden/latent)
- Observations (visible)
- Transition probabilities between states
- Emission probabilities of observations from states

5.5 Conditional Random Fields (CRF)

CRFs are discriminative models used for structured prediction. Unlike HMMs, they don't assume independence between observations and can incorporate arbitrary features.

CRFs are particularly effective for:
- Named entity recognition
- Part-of-speech tagging
- Chunking

They can capture complex dependencies and use rich feature sets.

Chapter 6: Deep Learning Foundations for NLP

6.1 Neural Networks Basics

Artificial neural networks are computing systems inspired by biological neural networks. They consist of:
- Input layer: Receives the input features
- Hidden layers: Process and transform the input
- Output layer: Produces the final prediction

Key concepts:
- Neurons/units: Basic computational units
- Weights and biases: Learnable parameters
- Activation functions: Non-linear transformations (ReLU, tanh, sigmoid)
- Backpropagation: Algorithm for computing gradients
- Gradient descent: Optimization algorithm for updating weights

6.2 Feedforward Neural Networks for Text

Feedforward networks can be used for text classification by:
1. Converting text to fixed-size vectors (e.g., averaging word embeddings)
2. Passing through hidden layers
3. Outputting class probabilities

Advantages:
- Can learn complex non-linear patterns
- Automatic feature learning

Limitations:
- Requires fixed-size input
- Doesn't capture sequential information well

6.3 Recurrent Neural Networks (RNN)

RNNs are designed to recognize patterns in sequences by maintaining a hidden state that captures information about previous elements.

At each time step t:
h_t = f(h_{t-1}, x_t)
y_t = g(h_t)

where h_t is the hidden state, x_t is the input, and y_t is the output.

Applications:
- Language modeling
- Text generation
- Sequence classification
- Machine translation

Challenges:
- Vanishing gradient problem: Difficulty learning long-term dependencies
- Exploding gradient problem: Gradients become too large
- Sequential computation: Cannot be easily parallelized

6.4 Long Short-Term Memory (LSTM)

LSTM is a special kind of RNN capable of learning long-term dependencies. It addresses the vanishing gradient problem through a gating mechanism.

LSTM components:
- Forget gate: Decides what information to discard from cell state
- Input gate: Decides what new information to store
- Output gate: Decides what to output based on cell state
- Cell state: Long-term memory that flows through the network

The gating mechanisms allow LSTMs to selectively remember or forget information, making them effective for tasks requiring long-range dependencies.

6.5 Gated Recurrent Units (GRU)

GRU is a simplified version of LSTM with fewer parameters. It combines the forget and input gates into a single update gate and merges the cell state and hidden state.

GRU components:
- Update gate: Controls how much of the previous hidden state to keep
- Reset gate: Controls how much of the previous hidden state to forget

GRUs are computationally more efficient than LSTMs and often perform comparably.

6.6 Bidirectional RNNs

Bidirectional RNNs process sequences in both forward and backward directions, allowing the model to have access to both past and future context.

Output at time t: y_t = g(h_forward_t, h_backward_t)

This is particularly useful for tasks where future context is available:
- Named entity recognition
- Part-of-speech tagging
- Sentence encoding

Chapter 7: Attention Mechanisms and Transformers

7.1 The Attention Mechanism

The attention mechanism allows models to focus on specific parts of the input when producing output. Instead of compressing all information into a fixed-size vector, attention computes a weighted sum of all input representations.

Attention score: Measures the relevance of each input to the current output
Attention weight: Normalized attention scores (using softmax)
Context vector: Weighted sum of input representations

Types of attention:
- Self-attention: Attention over the same sequence
- Cross-attention: Attention between two different sequences
- Multi-head attention: Multiple attention mechanisms in parallel

7.2 Transformer Architecture

Transformers revolutionized NLP by replacing recurrence with attention mechanisms. The original transformer consists of an encoder and decoder, each with multiple layers.

Encoder:
- Multi-head self-attention
- Position-wise feedforward network
- Layer normalization and residual connections

Decoder:
- Masked multi-head self-attention
- Multi-head cross-attention over encoder outputs
- Position-wise feedforward network
- Layer normalization and residual connections

Key innovations:
- Positional encoding: Adds position information to embeddings
- Parallel computation: Can process all positions simultaneously
- Self-attention: Captures long-range dependencies efficiently

7.3 Positional Encoding

Since transformers don't have inherent notion of sequence order, positional encodings are added to input embeddings to provide position information.

Original transformer uses sinusoidal positional encoding:
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Modern approaches also use learned positional embeddings.

7.4 Multi-Head Attention

Multi-head attention allows the model to attend to information from different representation subspaces.

Process:
1. Project queries, keys, and values multiple times (h times for h heads)
2. Compute attention for each head in parallel
3. Concatenate outputs from all heads
4. Apply final linear projection

This allows the model to capture different types of relationships simultaneously.

Chapter 8: Pre-trained Language Models

8.1 Transfer Learning in NLP

Transfer learning involves pre-training a model on a large corpus and then fine-tuning it for specific tasks. This approach has become the dominant paradigm in NLP.

Pre-training objectives:
- Language modeling: Predict next word
- Masked language modeling: Predict masked words
- Next sentence prediction: Determine if two sentences are consecutive

Benefits:
- Leverage large amounts of unlabeled data
- Require less task-specific labeled data
- Achieve better performance across tasks
- Faster convergence during fine-tuning

8.2 BERT (Bidirectional Encoder Representations from Transformers)

BERT uses bidirectional transformers to learn contextualized representations by training on two objectives:

Masked Language Modeling (MLM): Randomly mask 15% of tokens and predict them based on bidirectional context.

Next Sentence Prediction (NSP): Predict whether two sentences are consecutive in the original text.

BERT variants:
- RoBERTa: Optimized BERT with better pre-training strategies
- ALBERT: Parameter-efficient BERT with factorized embeddings
- DistilBERT: Smaller, faster version through knowledge distillation
- ELECTRA: Replaces MLM with replaced token detection

8.3 GPT (Generative Pre-trained Transformer)

GPT uses autoregressive language modeling, predicting the next token given all previous tokens. Unlike BERT, it's unidirectional (left-to-right).

Evolution:
- GPT-1: 117M parameters
- GPT-2: 1.5B parameters
- GPT-3: 175B parameters
- GPT-4: Multimodal capabilities

GPT excels at:
- Text generation
- Few-shot learning
- In-context learning
- Creative writing

8.4 T5 (Text-to-Text Transfer Transformer)

T5 frames all NLP tasks as text-to-text problems, using the same model, objective, and decoding procedure for all tasks.

Examples:
- Translation: "translate English to German: Hello" → "Hallo"
- Summarization: "summarize: [long text]" → "[summary]"
- Classification: "sentiment: This movie is great!" → "positive"

This unified framework simplifies multi-task learning and transfer learning.

8.5 Other Notable Models

XLNet: Combines benefits of autoregressive and autoencoding models through permutation language modeling.

BART: Combines bidirectional encoder with autoregressive decoder, effective for generation tasks.

mBART, mT5: Multilingual versions trained on multiple languages simultaneously.

DeBERTa: Enhanced BERT with disentangled attention mechanism.

Chapter 9: Sentiment Analysis and Opinion Mining

9.1 Introduction to Sentiment Analysis

Sentiment analysis determines the emotional tone behind text. It's crucial for:
- Social media monitoring
- Customer feedback analysis
- Brand reputation management
- Market research
- Political opinion tracking

Granularity levels:
- Document level: Overall sentiment of entire document
- Sentence level: Sentiment of individual sentences
- Aspect level: Sentiment toward specific aspects/features

9.2 Approaches to Sentiment Analysis

Lexicon-based approaches:
- Use predefined dictionaries of words with sentiment scores
- Simple and interpretable
- Examples: VADER, SentiWordNet, AFINN
- Limitations: Cannot handle context, sarcasm, or domain-specific language

Machine Learning approaches:
- Train classifiers on labeled datasets
- Features: BoW, TF-IDF, n-grams, POS tags
- Algorithms: Naive Bayes, SVM, Logistic Regression
- Better at handling context and domain-specific language

Deep Learning approaches:
- RNNs, LSTMs, CNNs for sequence modeling
- Pre-trained models (BERT, RoBERTa) fine-tuned for sentiment
- Can capture complex patterns and contextual information
- State-of-the-art performance

9.3 Challenges in Sentiment Analysis

Sarcasm and Irony: "Great, another rainy day!" - Literally positive but actually negative.

Negation handling: "not good" should be negative, not positive.

Context dependency: "This film is sick!" - Positive in slang, negative literally.

Aspect-based sentiment: "The food was excellent but the service was terrible" - Mixed sentiments.

Domain adaptation: Sentiment expressions vary across domains (movie reviews vs. product reviews).

Multimodal sentiment: Combining text with images, videos, or audio for richer analysis.

9.4 Aspect-Based Sentiment Analysis

ABSA identifies specific aspects of entities and determines sentiment toward each aspect.

Steps:
1. Aspect extraction: Identify aspects mentioned in text
2. Opinion extraction: Identify opinion expressions
3. Aspect-sentiment pairing: Link opinions to aspects
4. Sentiment classification: Determine polarity for each aspect

Example:
"The battery life is amazing but the screen is too small."
- battery life: positive
- screen: negative

9.5 Emotion Detection

Beyond positive/negative/neutral, emotion detection identifies specific emotions like:
- Joy, sadness, anger, fear, surprise, disgust (Ekman's basic emotions)
- More fine-grained emotions: excitement, frustration, confusion, etc.

Applications:
- Mental health monitoring
- Customer service quality assessment
- Content recommendation
- Educational technology

Chapter 10: Machine Translation

10.1 Evolution of Machine Translation

Rule-based MT (RBMT): Uses linguistic rules and bilingual dictionaries. Requires extensive manual rule creation.

Statistical MT (SMT): Learns translation models from parallel corpora using statistical methods. Includes phrase-based and hierarchical approaches.

Neural MT (NMT): Uses neural networks, typically encoder-decoder architectures with attention. Current state-of-the-art approach.

10.2 Neural Machine Translation Architecture

Basic encoder-decoder:
1. Encoder: Processes source language into a context vector
2. Decoder: Generates target language from context vector

With attention:
1. Encoder: Produces sequence of hidden states
2. Attention: Computes context vector for each decoding step
3. Decoder: Uses attention context to generate each target word

Transformer-based NMT:
- Uses self-attention instead of recurrence
- Parallel processing enables faster training
- Better at capturing long-range dependencies
- Examples: MarianMT, mBART, NLLB

10.3 Challenges in Machine Translation

Ambiguity: Words with multiple meanings need context to translate correctly.

Idioms and expressions: Literal translation often doesn't work.

Low-resource languages: Limited parallel data available for training.

Morphologically rich languages: Complex word forms and inflections.

Cultural context: Some concepts don't have direct equivalents.

Named entities: Proper nouns may need transliteration rather than translation.

10.4 Evaluation Metrics

BLEU (Bilingual Evaluation Understudy): Measures n-gram overlap between machine translation and reference translations.

METEOR: Considers synonyms, stemming, and paraphrasing.

ROUGE: Originally for summarization, also used for translation.

chrF: Character-level F-score, better for morphologically rich languages.

BERTScore: Uses contextual embeddings to measure semantic similarity.

Human evaluation: Still the gold standard for quality assessment.

10.5 Multilingual Models

Models trained on multiple languages simultaneously can:
- Share knowledge across languages
- Enable zero-shot translation (translate between language pairs not seen during training)
- Improve performance on low-resource languages

Examples:
- mBART: Multilingual sequence-to-sequence model
- mT5: Multilingual text-to-text transformer
- NLLB (No Language Left Behind): Supports 200+ languages

Chapter 11: Question Answering Systems

11.1 Types of Question Answering

Extractive QA: Extract answer spans directly from a given context passage.
Example: SQuAD (Stanford Question Answering Dataset)

Generative QA: Generate answers based on understanding, not limited to exact text spans.
Example: Natural Questions, ELI5 (Explain Like I'm 5)

Open-domain QA: Answer questions without a pre-specified context, requiring information retrieval.

Closed-domain QA: Answer questions within a specific domain with structured knowledge.

11.2 Extractive Question Answering

Models predict start and end positions of the answer span in the context.

Architecture:
1. Encode question and context together
2. Compute start and end probabilities for each position
3. Select span with highest joint probability

Popular models:
- BERT-based models
- RoBERTa
- ALBERT
- ELECTRA

Training requires:
- Context passages
- Questions
- Answer spans with start/end positions

11.3 Generative Question Answering

Models generate free-form answers, potentially synthesizing information from multiple sources.

Approaches:
- Sequence-to-sequence models
- T5: Treats QA as text-to-text generation
- BART: Encoder-decoder architecture
- GPT: Autoregressive generation

Advantages:
- More flexible answer formats
- Can synthesize information
- Natural conversational responses

Challenges:
- May generate incorrect or hallucinated information
- Harder to evaluate
- Requires more computational resources

11.4 Retrieval-Augmented Generation (RAG)

RAG combines retrieval and generation to leverage external knowledge:

Process:
1. Retrieve relevant documents from a knowledge base using dense or sparse retrieval
2. Encode question and retrieved documents
3. Generate answer conditioned on question and retrieved context

Benefits:
- Access to vast external knowledge
- More factual and grounded responses
- Can cite sources
- Updatable knowledge without retraining

Components:
- Dense retriever: Uses embeddings to find relevant documents (DPR, ColBERT)
- Generator: Produces answer from retrieved context (BART, T5, GPT)

11.5 Conversational QA

Multi-turn question answering where context builds over multiple exchanges.

Challenges:
- Coreference resolution across turns
- Context management
- Maintaining conversation history
- Handling follow-up questions

Examples:
- CoQA (Conversational Question Answering)
- QuAC (Question Answering in Context)

Chapter 12: Text Summarization

12.1 Types of Summarization

Extractive Summarization:
- Selects important sentences from source document
- Preserves original wording
- Generally more factually accurate
- May lack coherence

Abstractive Summarization:
- Generates new sentences
- More human-like summaries
- Better coherence
- Risk of generating incorrect information

Single-document vs. Multi-document:
- Single: Summarize one document
- Multi: Synthesize information from multiple sources

12.2 Extractive Summarization Techniques

Graph-based methods:
- TextRank: Ranks sentences using PageRank algorithm
- LexRank: Similar to TextRank with different similarity metric

Feature-based methods:
- Score sentences based on features (position, length, keywords, etc.)
- Use classification or regression to select sentences

Clustering-based methods:
- Cluster similar sentences
- Select representative sentences from each cluster

Neural extractive models:
- BERT-based sentence scoring
- Hierarchical encoders for document structure
- Reinforcement learning for sentence selection

12.3 Abstractive Summarization

Sequence-to-sequence models:
- Encoder: Processes source document
- Decoder: Generates summary

Attention mechanisms:
- Focus on relevant parts of source
- Copy mechanism: Directly copy rare words
- Coverage mechanism: Avoid redundancy

Transformer-based models:
- BART: Pre-trained encoder-decoder
- T5: Text-to-text framework
- Pegasus: Pre-trained specifically for summarization
- BERT2BERT: Use BERT as both encoder and decoder

12.4 Evaluation Metrics

ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
- ROUGE-N: N-gram overlap
- ROUGE-L: Longest common subsequence
- ROUGE-S: Skip-bigram overlap

BLEU: Originally for translation, also used for summarization

BERTScore: Semantic similarity using contextual embeddings

Factuality metrics:
- Check consistency with source
- Identify hallucinations

Human evaluation:
- Fluency: Grammatical correctness
- Coherence: Logical flow
- Relevance: Importance of content
- Consistency: Factual accuracy

12.5 Challenges and Future Directions

Faithfulness: Ensuring summaries don't contain false information

Long document summarization: Handling documents longer than model's context window

Controllable summarization: Generating summaries with specific length, style, or focus

Multi-document summarization: Handling redundancy and contradictions

Domain adaptation: Adapting to specialized domains (medical, legal, scientific)

Chapter 13: Information Extraction

13.1 Named Entity Recognition (NER)

NER identifies and classifies named entities in text.

Standard entity types:
- Person (PER)
- Organization (ORG)
- Location (LOC)
- Date/Time
- Money
- Percent

Domain-specific entities:
- Medical: Diseases, symptoms, drugs
- Legal: Laws, courts, case citations
- Scientific: Chemical compounds, proteins, species

Approaches:
- Rule-based: Pattern matching, gazetteers
- Machine learning: CRF, SVM with linguistic features
- Deep learning: BiLSTM-CRF, BERT-based models

Challenges:
- Entity boundary detection
- Nested entities
- Ambiguous entities
- Domain adaptation

13.2 Relation Extraction

Identifies relationships between entities in text.

Example:
"Steve Jobs founded Apple in 1976."
Relation: (Steve Jobs, founded, Apple)

Types:
- Binary relation extraction: Relationships between two entities
- N-ary relation extraction: Multiple entities and relations
- Open information extraction: Extract relations without predefined schema

Approaches:
- Pattern-based: Hand-crafted or learned patterns
- Supervised: Train classifiers on labeled entity pairs
- Distant supervision: Use knowledge bases for weak supervision
- Neural models: Encode entity context for relation classification

13.3 Event Extraction

Identifies events mentioned in text along with participants and attributes.

Event components:
- Event trigger: Word indicating the event
- Event arguments: Entities participating in the event
- Event type: Category of the event

Example:
"Apple acquired a startup for $100 million."
Event type: Acquisition
Acquirer: Apple
Acquired: startup
Price: $100 million

Applications:
- News analysis
- Financial analysis
- Medical records processing
- Social media monitoring

13.4 Knowledge Graph Construction

Combines NER, relation extraction, and entity linking to build structured knowledge graphs.

Process:
1. Extract entities from text
2. Extract relations between entities
3. Link entities to knowledge base
4. Resolve duplicates and conflicts
5. Add to knowledge graph

Knowledge graphs:
- Nodes: Entities
- Edges: Relations
- Properties: Attributes

Examples: Google Knowledge Graph, Wikidata, DBpedia

Chapter 14: Dialogue Systems and Chatbots

14.1 Types of Dialogue Systems

Task-oriented dialogue:
- Specific goal (booking, reservations, customer service)
- Structured conversations
- Need understanding of user intent and slots
- Examples: Restaurant booking, flight reservations

Open-domain dialogue:
- General conversation
- No specific goal
- More challenging
- Examples: Social chatbots, virtual companions

Question answering dialogue:
- Focused on answering questions
- May involve multi-turn interactions
- Examples: Customer support, educational assistants

14.2 Components of Dialogue Systems

Natural Language Understanding (NLU):
- Intent classification: What does the user want?
- Slot filling: Extract relevant information
- Example: "Book a table for 2 at 7pm"
  Intent: book_table
  Slots: {num_people: 2, time: 7pm}

Dialogue Management:
- Track conversation state
- Decide next action
- Handle context and history
- Manage clarifications and errors

Natural Language Generation (NLG):
- Generate appropriate responses
- Maintain consistent tone
- Provide informative feedback

14.3 Neural Dialogue Models

Retrieval-based models:
- Select response from predefined set
- More controlled and safe
- Limited flexibility

Generative models:
- Generate responses from scratch
- More flexible and diverse
- Risk of inappropriate or incorrect responses

Hybrid approaches:
- Combine retrieval and generation
- Use retrieval for safe fallback
- Generate for more natural responses

End-to-end models:
- Learn directly from conversation data
- No explicit module separation
- Examples: Transformer-based models, GPT for dialogue

14.4 Evaluation of Dialogue Systems

Automatic metrics:
- Perplexity: Language model quality
- BLEU, ROUGE: Response similarity
- Distinct-n: Response diversity
- BERTScore: Semantic similarity

Human evaluation:
- Appropriateness: Is response relevant?
- Fluency: Is response well-formed?
- Informativeness: Does it provide useful information?
- Engagingness: Is conversation interesting?

Task success metrics:
- Task completion rate
- Number of turns
- User satisfaction

14.5 Challenges in Dialogue Systems

Context tracking: Maintaining conversation history across turns

Consistency: Avoiding contradictions in multi-turn dialogues

Personality: Maintaining consistent personality and tone

Safety: Avoiding toxic, biased, or harmful responses

Grounding: Staying factual and avoiding hallucinations

Multi-modal understanding: Integrating vision, speech, etc.

Chapter 15: Ethics and Bias in NLP

15.1 Types of Bias in NLP

Training data bias:
- Historical bias: Reflecting past societal biases
- Representation bias: Under-representing certain groups
- Measurement bias: Proxy variables introducing bias

Model bias:
- Amplification: Amplifying biases present in data
- Spurious correlations: Learning wrong associations
- Fairness across groups: Different performance for different demographics

Deployment bias:
- Feedback loops: Model predictions affecting future data
- Context mismatch: Different deployment context than training

15.2 Ethical Concerns

Privacy: Processing personal information in text

Transparency: Black-box models difficult to interpret

Accountability: Who is responsible for model mistakes?

Dual use: Technology used for harmful purposes

Environmental impact: Large models require significant energy

Access and inclusion: Ensuring benefits reach all communities

15.3 Bias Detection and Mitigation

Detection methods:
- Bias evaluation datasets
- Counterfactual evaluation
- Embedding association tests
- Downstream task performance across groups

Mitigation strategies:
- Data augmentation and balancing
- Adversarial debiasing
- Fair representation learning
- Post-processing corrections
- Careful prompt engineering

Limitations:
- Trade-offs between fairness metrics
- Bias may transfer across tasks
- Context-dependent definitions of fairness

15.4 Responsible AI Practices

Documentation:
- Dataset datasheets
- Model cards
- Impact assessments

Testing and validation:
- Diverse test sets
- Adversarial testing
- Red teaming

Monitoring and auditing:
- Continuous performance monitoring
- Regular bias audits
- User feedback mechanisms

Inclusive development:
- Diverse development teams
- Stakeholder involvement
- Community engagement

Chapter 16: Future Trends and Conclusion

16.1 Emerging Trends in NLP

Multimodal models:
- Combining text, images, video, audio
- Vision-language models (CLIP, DALL-E)
- Speech and text integration

Few-shot and zero-shot learning:
- Learning from minimal examples
- Prompt-based learning
- In-context learning

Efficient models:
- Model compression and distillation
- Sparse models and mixture of experts
- Quantization and pruning

Multilingual and cross-lingual:
- Better support for low-resource languages
- Cross-lingual transfer
- Massively multilingual models

Reasoning and commonsense:
- Improved logical reasoning
- Better commonsense understanding
- Integration with knowledge graphs

16.2 Open Challenges

Long-context understanding:
- Processing very long documents
- Maintaining coherence across long contexts

Factual grounding:
- Reducing hallucinations
- Verifying claims
- Attribution and citations

Domain adaptation:
- Adapting to specialized domains
- Continual learning
- Transfer across tasks and domains

Robustness:
- Adversarial robustness
- Out-of-distribution generalization
- Handling noisy and informal text

Interpretability:
- Understanding model decisions
- Providing explanations
- Building trust

16.3 Conclusion

Natural Language Processing has made remarkable progress in recent years, transforming from rule-based systems to sophisticated neural models that can understand and generate human-like text. The field continues to evolve rapidly, with new models, techniques, and applications emerging constantly.

Key takeaways:
- Deep learning and transformers have revolutionized NLP
- Pre-trained models enable transfer learning across tasks
- Attention mechanisms capture long-range dependencies
- Ethical considerations are crucial for responsible AI
- Multimodal and multilingual capabilities are expanding
- Many challenges remain, driving ongoing research

The future of NLP promises even more exciting developments as models become more capable, efficient, and accessible. As the technology matures, it will continue to find applications in diverse domains, improving how humans interact with computers and access information.

Success in NLP requires:
- Strong foundations in machine learning and linguistics
- Understanding of classical and modern techniques
- Awareness of ethical implications
- Continuous learning as the field evolves
- Practical experience with real-world applications

Whether you're building chatbots, analyzing sentiment, translating languages, or extracting information, the principles and techniques covered in this guide provide a solid foundation for working with natural language processing. The field offers tremendous opportunities for innovation and impact, and we encourage you to explore, experiment, and contribute to this exciting area of artificial intelligence.
"""

def create_nlp_pdf():
    """Create a PDF document with NLP content."""
    
    # Create PDF
    pdf_path = r"C:\Users\HP\Downloads\NLP_Book.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER, fontSize=16, spaceAfter=12))
    
    # Split content into paragraphs
    paragraphs = nlp_content.strip().split('\n\n')
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Check if it's a title/chapter heading
        if para.startswith('Chapter') or para == 'Natural Language Processing (NLP)' or para == 'Conclusion':
            p = Paragraph(f'<b>{para}</b>', styles['Center'])
            elements.append(p)
            elements.append(Spacer(1, 0.2*inch))
        # Check if it's a section heading (starts with number)
        elif para[0].isdigit() and '.' in para[:5]:
            p = Paragraph(f'<b>{para}</b>', styles['Heading2'])
            elements.append(p)
            elements.append(Spacer(1, 0.1*inch))
        else:
            # Regular paragraph
            p = Paragraph(para, styles['Justify'])
            elements.append(p)
            elements.append(Spacer(1, 0.15*inch))
    
    # Build PDF
    doc.build(elements)
    print(f"✅ PDF created successfully: {pdf_path}")

if __name__ == "__main__":
    try:
        create_nlp_pdf()
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        print("\nInstall required package:")
        print("pip install reportlab")
