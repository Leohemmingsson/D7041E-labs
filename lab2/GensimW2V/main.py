import gensim
import logging
import numpy as np
import nltk
import help_functions as hf
from collections import defaultdict

# Ensure the 'wordnet' resource is available
nltk.download("wordnet")

# Set up logging
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

# Create a lemmatizer
lemmatizer = nltk.WordNetLemmatizer()

# Read the file and create a list containing all sentences found in the text
sentences = []
with open("lemmatized.text", "r") as file:
    for line in file:
        sentences.append(line.split())

# Define different dimensionalities to test
dimensionalities = [10, 100, 300]
num_simulations = 5  # Number of simulations to run for each dimensionality
threshold = 0.00055  # Threshold for downsampling

# Dictionary to hold the accuracies for each dimensionality
accuracy_results = defaultdict(list)

# Perform simulations for each dimensionality
for dimension in dimensionalities:
    for simulation in range(num_simulations):
        # Create a Word2Vec model with the current dimensionality
        model = gensim.models.Word2Vec(
            sentences, min_count=1, sample=threshold, sg=1, vector_size=dimension
        )

        i = 0  # Counter for TOEFL tests
        right_answers = 0  # Variable for correct answers
        number_skipped_tests = (
            0  # Tests could be skipped if words are not in the vocabulary
        )

        # Open the TOEFL test file
        with open("new_toefl.txt", "r") as text_file:
            while i < 80:
                line = text_file.readline()
                if not line:
                    break  # Exit if no more lines

                words = line.split()
                # Lemmatize words in the current test
                words = [
                    lemmatizer.lemmatize(
                        lemmatizer.lemmatize(lemmatizer.lemmatize(word, "v"), "n"), "a"
                    )
                    for word in words
                ]

                vectors = []
                if (
                    words[0] in model.wv
                ):  # Check if there is an embedding for the query word
                    vectors.append(model.wv[words[0]])
                    for k in range(1, 5):
                        if words[k] in model.wv:
                            vectors.append(model.wv[words[k]])
                        else:
                            vectors.append(
                                np.random.randn(dimension)
                            )  # Assign random vector

                    # Calculate and store the result for the current TOEFL test
                    right_answers += hf.get_answer_mod(vectors)
                else:
                    number_skipped_tests += 1  # Skip test if no embedding

                i += 1

        # Calculate the percentage of correct answers
        if (80 - number_skipped_tests) == 0:
            continue  # Avoid division by zero
        percentage_correct = 100 * right_answers / (80 - number_skipped_tests)
        accuracy_results[dimension].append(percentage_correct)
        print(
            f"Simulation {simulation + 1}/{num_simulations}, Dimension: {dimension}, Accuracy: {percentage_correct}%"
        )

# Report the accuracies for each dimensionality
for dimension, accuracies in accuracy_results.items():
    print(f"Dimensionality: {dimension}, Accuracies over simulations: {accuracies}")

# Analyze how accuracy changes with dimensionality
# Your analysis code goes here
