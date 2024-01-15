import nltk
import string
import math
import numpy as np
import networkx as nx

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance



'''''''''''''''''''''''''''''''''
READ AND RETURN ALL DOCUMENTS
'''''''''''''''''''''''''''''''''
def read_documents(file_paths):
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            document = file.read()
            documents.append(document)
    return documents


'''''''''''''''''''''''''''''''''
PREPROCESS TEXT IN DOCUMENT
'''''''''''''''''''''''''''''''''
def preprocess_text(text):
    preprocessed_sentences = []

    # Split the text into sentences
    sentences = sent_tokenize(text) 
    
    # Iterate through each sentence in the text, tokenize each word in the sentence and replace non-alphabetic characters with a space
    preprocessed_sentences =  [word_tokenize(sentence.replace("[^a-zA-Z]", " ")) for sentence in sentences]

    return preprocessed_sentences


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
CALCULATE SIMILARITY BETWEEN TWO SENTENCES USING COSINE SIMILARITY AND BAG-OF-WORDS APPROACH
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def sentence_similarity(sentence1,sentence2,stopwords=None):
    if stopwords is None:
        stopwords=[]

    # Convert each word in the sentences to lowercase.
    sentence1 = [word.lower() for word in sentence1] 
    sentence2 = [word.lower() for word in sentence2]

    # Create a list of all unique words present in both sentences.
    all_words = list(set(sentence1+sentence2)) 

    # Initialize vectors of zeros with a length equal to the number of unique words. These vectors will represent the word frequencies in each sentence.
    word_frequency_vector1 = [0] * len(all_words) 
    word_frequency_vector2 = [0] * len(all_words)

    # Build the vector for the first sentence.
    for word in sentence1:
        if word in stopwords or word in string.punctuation:
            continue
        word_frequency_vector1[all_words.index(word)] += 1 # if its not a stopwords or a punctuation, the word frequency is incremented in the corresponding vector.
    # Build the vector for the second sentence.
    for word in sentence2:
        if word in stopwords or word in string.punctuation:
            continue
        word_frequency_vector2[all_words.index(word)] += 1

    # Calculate similarity using cosine distance and return the result.
    return 1-cosine_distance(word_frequency_vector1, word_frequency_vector2)


'''''''''''''''''''''''''''''''''
BUILD A SIMILARITY MATRIX
'''''''''''''''''''''''''''''''''
def gen_sim_matrix(sentences, stop_words):
    # Create an empty similarity matrix with dimensions (number of sentences) x (number of sentences).
    similarity_matrix = np.zeros((len(sentences),len(sentences)))
    
    # Iterate through all pairs of sentences and calculate their similarity.
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            # Skip calculating similarity for the same sentence
            if idx1 == idx2:
                continue
            # Calculate the similarity between sentences[idx1] and sentences[idx2]. The result is stored in the corresponding element of the similarity matrix.
            similarity_matrix[idx1][idx2]= sentence_similarity(sentences[idx1],sentences[idx2],stop_words)

    return similarity_matrix


'''''''''''''''''''''''''''''''''
GENERATE A SUMMARY
'''''''''''''''''''''''''''''''''
def generate_summary(document, top_n=5):
    summarize_text=[]
    stop_words = stopwords.words('english')

    # Preprocess the text of the document.
    sentences = preprocess_text(document)

    # Calculate the similarity matrix between sentences.
    sentence_similarity_matrix = gen_sim_matrix(sentences,stop_words)
    # Create a graph from the similarity matrix where nodes represent sentences, and edges represent the similarity between sentences.
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix) 

    # Calculate PageRank scores for each node (sentence) in the graph. Sentences that are more similar to others or act as central points in the similarity graph receive higher PageRank scores.
    scores = nx.pagerank(sentence_similarity_graph) 
    # Sort sentences based on PageRank scores, from highest to lowest.
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)
    # Select the top sentences based on the specified number (top_n).
    for i in range(min(top_n, len(ranked_sentence))):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    

    # Concatenate the selected sentences to form the "final" summary.
    return ' '.join(summarize_text)


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
COLLATE THE SUMMARIES IN DOCUMENTS BASED ON THE NUMBER OF STEP
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def collate_summaries(summaries, num_steps):
    
    # Determine how many summaries will be grouped together at each step
    len_summaries = len(summaries)
    step_size = math.ceil(len_summaries / num_steps)

    collated_summaries = []
    collated_summary_temp =[]
    n =0

    print("\n\n STEP: ",num_steps)

    # Iterate through the summaries and group them based on the step size
    while n < len_summaries:
        for i in range(step_size):
            if n < len(summaries):
                # Append the current summary to the temporary list
                collated_summary_temp.append(summaries[n])
                n+=1
        # Join the summaries in the temporary list to create a collated summary for the current step
        collated_summary = ' '.join(collated_summary_temp)
        print("\nDocument ",n-step_size+1,":\n", collated_summary)

        # Append the collated summary to the list of collated summaries
        collated_summaries.append(collated_summary)
        # Reset the temporary list for the next iteration
        collated_summary_temp=[]

    
    return collated_summaries


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
GENERATE THE SUMMARY OF THE SUMMARIES RECURSIVELY UNTIL THE NUMBER OF STEPS IS EXHAUSTED
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def recursive_summarization(documents, summary_size, num_steps):
    # Base case: If the number of steps is 0, return the current set of documents
    if num_steps == 0:
        return documents

    # Generate summaries for each document in the current set
    summaries = [generate_summary(doc, summary_size) for doc in documents]
    # Collate the summaries to create a new set of documents
    documents = collate_summaries(summaries, num_steps)
    # Recursively call the function with the new set of documents and reduced number of steps
    return recursive_summarization(documents, summary_size, num_steps - 1)




'''''''''''''''''''''''''''''''''
EXAMPLE OF USE
'''''''''''''''''''''''''''''''''
file_paths = ['document1.txt', 'document2.txt', 'document3.txt']
documents = read_documents(file_paths)

#half number of sentences required
#Example: for final summary with 4 sentences long, enter 2
summary_size = 2
num_steps = 3

final_summaries = recursive_summarization(documents, summary_size, num_steps)
print('\n\n')
for i, summary in enumerate(final_summaries):
    print(f'\nFINAL SUMMARY {i + 1}:\n{summary}\n')
