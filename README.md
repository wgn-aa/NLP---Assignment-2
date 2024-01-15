# Project

Here is the code with commentary for the second Natural Language Processing (NLP) assignment



## Subject

Large Language Models can be trained for a number of purposes. In this assignment, candidates are required to implement a hierarchical system for summarization in the same system. The implementation are to be either in Python, while using NLTK, or in Java while using OpenNLP.


The method will be based on the concept of giving as input to ChatGPT a set of N  documents, each constituting a part of another given document, that we assume to be sufficiently independent to each other to be valued as separate. The set, overall, is formed in a way that exceeds the context window. In input it is given the size of the summary for each step and the number of steps in the summarization process. The pipeline will be 

* Generate a summary for each document of the given size;
* Collate the summaries in documents based on the number of steps;
* Generate the summary of the summaries recursively until the number of steps is exhausted.

The assignment does not require the evaluation of the performances.



##  Operational Structure:

* Input Documents:
    * The system takes a set of input documents in the form of text files (file_paths variable), the size of the summary for each step and the number of steps

* Text Preprocessing:
    * The text within each document undergoes preprocessing using the preprocess_text function.
    * Tokenization: The text is tokenized into sentences, and each sentence is further tokenized into words.
    * Non-alphabetic characters are replaced with spaces.

* Similarity Matrix Generation:
    * A similarity matrix is generated using cosine similarity between pairs of sentences in the documents.
    * The gen_sim_matrix function creates an empty similarity matrix and fills it with cosine similarity scores.

* Graph Construction:
    * A graph is constructed from the similarity matrix using NetworkX.
    * Nodes in the graph represent sentences, and edges represent the similarity between sentences.

* PageRank Scores Calculation:
    * PageRank scores are calculated for each node (sentence) in the graph using NetworkX's pagerank function.
    * Sentences that are more similar to others or act as central points in the similarity graph receive higher PageRank scores.

* Summary Generation:
    * The sentences are ranked based on their PageRank scores, and the top sentences are selected to form a summary.
    * The generate_summary function takes care of this process, allowing customization of the summary size.

* Recursive Summarization:
    * The recursive_summarization function is used to iteratively generate summaries.
    * At each iteration, summaries are collated, and the process is repeated until the specified number of steps is reached.


* Output Summaries:
    * The summaries at each step are print after been collated.
    * The final summaries are printed or can be stored for further use.



## Pipeline:

* Initialization:
    * Import necessary libraries and modules (NLTK, NumPy, NetworkX).

* Document Reading:
    * Read a set of N independent documents from file paths using the read_documents function. Each documents constituting a part of a larger document and are assumed to be sufficiently independent.

* Text Summarization Iterations:
    * Define parameters such as summary_size and num_steps.
    * Use recursive_summarization to iteratively generate summaries.
    * At each step, the following process is repeated:
        * Generate a Summary for Each Document:
            * Use the generate_summary function to generate a summary of the specified size for each document.
        * Collate the Summaries:
            * Use collate_summaries to collate the generated summaries into new documents based on the number of steps.
        * Update Input Documents:
            * Replace the original set of documents with the collated summaries for the next iteration.

* Output:
    * Print or store the final set of summaries.



## Technologies Used:

* Programming Language: **Python**
* Libraries: **NLTK**,**networkx**, **numpy**



## To test a new document

If you want to test a new large document summary, simply:

* separate your document into several parts according to its size
* put each part in "document[i].txt" or create new .txt files.
* put the file names in the file_paths variable, or create a new one in the form of a list.
* specify desired summary size and number of steps
* run the program

You'll then be able to see the summary of your document

