import random
import math
from tokenizer import encode, idx_to_char, char_to_idx

def initialize_weights(vocab_size, embed_size):
    """
    Create random starting weights for our model.\n
    vocab_size: how many different characters we have (27)([a-z ])\n
    embed_size: size of the internal representation
      (how many numbers we use to represent each character) (32)\n
    Why not start with zeros? If all weights were the same (like all zeros),
    every neuron (processing unit) would learn the exact same thing.\n
    Random values break this symmetry so different parts learn different patterns.
    """

    # Embedding: converts each character number into embed_size numbers
    # Remember: each character is represented by a vector of numbers
    # and all of these numbers are random at first.
    # This is where we create those random vectors of numbers.
    embedding = [[random.gauss(0, 0.1) for _ in range(embed_size)]
                 for _ in range(vocab_size)]

    # Attention weights: helps model focus on important previous characters
    W_query = [[random.gauss(0, 0.1) for _ in range(embed_size)]
               for _ in range(embed_size)]
    W_key = [[random.gauss(0, 0.1) for _ in range(embed_size)]
             for _ in range(embed_size)]
    W_value = [[random.gauss(0, 0.1) for _ in range(embed_size)]
               for _ in range(embed_size)]

    # Output layer: converts back to character predictions
    W_output = [[random.gauss(0, 0.1) for _ in range(vocab_size)]
                for _ in range(embed_size)]

    return {
        'embedding': embedding,
        'W_query': W_query,
        'W_key': W_key,
        'W_value': W_value,
        'W_output': W_output
    }

# Matrix A (2x3):          Matrix B (3x2):
# [1, 2, 3]                [1, 4]
# [4, 5, 6]                [2, 5]
#                          [3, 6]

# Result (2x2):
# [1×1 + 2×2 + 3×3, 1×4 + 2×5 + 3×6]  =  [14, 32]
# [4×1 + 5×2 + 6×3, 4×4 + 5×5 + 6×6]  =  [32, 77]
# https://www.mathsisfun.com/algebra/matrix-multiplying.html
def matmul(A, B):
    """Multiply two matrices (lists of lists).\n
    A: list of lists, shape (m x n)\n
    B: list of lists, shape (n x p)\n
    Returns: list of lists, shape (m x p)
    """
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

def softmax(values):
    """
    Convert a list of numbers into probabilities that sum to 1.
    Steps:
    - Make all numbers positive: Use e^x (exponential function)
    [2.3, -1.5, 4.1, 0.8] → [10.0, 0.22, 60.3, 2.2]
    - Make them sum to 1: Divide each by the total
    Total = 72.72
    [10.0/72.72, 0.22/72.72, 60.3/72.72, 2.2/72.72] = [0.14, 0.003, 0.83, 0.03]"""

    exp_values = [math.exp(v - max(values)) for v in values]
    sum_exp = sum(exp_values)

    return [v / sum_exp for v in exp_values]

def forward_pass(input_sequence, weights):
    """
    Take input characters and predict next character.
    Returns probabilities for each possible next character.
    """

    # Determine the embedding size from the first weight matrix.
    # All the embedding vectors have the same size, so we can just use the first one.
    # This helps us know how many numbers represent each character.
    embed_size = len(weights['embedding'][0])
    vocab_size = len(weights['W_output'][0])

    # Step 1: Convert each character in the input sequence to its embedding.
    # Each character index is replaced by its corresponding embedding vector.
    embeddings = [weights['embedding'][idx] for idx in input_sequence]

    # Step 2: Apply simple attention (averaged for simplicity).
    # Attention = Deciding which previous characters matter most.
    # We will just average the embeddings here for simplicity.
    context = [sum(col) / len(embeddings) for col in zip(*embeddings)]

    # Step 3: Project to output space
    output = [
      sum(context[i] * weights['W_output'][i][j] for i in range(embed_size))
        for j in range(vocab_size)
    ]

    # Step 4: Convert to probabilities
    probabilities = softmax(output)

    return probabilities

def cross_entropy_loss(probabilities, target_idx):
    """
    Calculate how wrong our prediction was.
    Lower loss = better prediction
    probabilities: our model's output for all characters
    target_idx: the correct character's index
    """
    # Loss is negative log of the probability we gave to the correct answer
    # If we gave 90% to correct answer: loss is low (good!)
    # If we gave 10% to correct answer: loss is high (bad!)
    return -math.log(probabilities[target_idx] + 1e-10)  # +1e-10 avoids log(0)

# This is where the learning happens.
# We adjust weights based on how wrong we were.
# It can be adaptive, using techniques like Adam or RMSProp for better results.
# We could also do adaptive learning rate based on confidence in wrong answer
# If we were very confident and wrong: learn faster
# If we were uncertain: learn slower
# With a fixed learning rater like 0.01, we just take small steps each time.
# Large steps can overshoot the best weights.
# After it learns the patterns well, those big steps make it "bounce around" and get worse.
# Like trying to land on a target but jumping too far each time.
# The learning can be made dynamic and complex, but here is a very simplified version.
def backward_pass(input_sequence, target_idx, weights, learning_rate=0.01):
    """
    Calculate gradients (how much to change each weight) and update weights.
    learning_rate: how big of a step to take (0.01 = small careful steps)
    """
    embed_size = len(weights['embedding'][0])

    # Forward pass to get prediction
    embeddings = [weights['embedding'][idx] for idx in input_sequence]
    context = [sum(col) / len(embeddings) for col in zip(*embeddings)]

    output = [sum(context[i] * weights['W_output'][i][j]
                  for i in range(embed_size))
              for j in range(len(weights['W_output'][0]))]

    probs = softmax(output)

    # Calculate error: difference between prediction and target
    output_error = probs.copy()
    output_error[target_idx] -= 1  # Gradient of cross-entropy loss

    # Update W_output: adjust based on how wrong we were
    for i in range(embed_size):
        for j in range(len(weights['W_output'][0])):
            gradient = context[i] * output_error[j]
            weights['W_output'][i][j] -= learning_rate * gradient

    # Update embeddings: push them in the right direction
    context_error = [sum(output_error[j] * weights['W_output'][i][j]
                         for j in range(len(output_error)))
                     for i in range(embed_size)]

    for _, idx in enumerate(input_sequence):
        for i in range(embed_size):
            gradient = context_error[i] / len(input_sequence)
            weights['embedding'][idx][i] -= learning_rate * gradient

    return probs

# Example usage:
if __name__ == "__main__":
    weights = initialize_weights(vocab_size=27, embed_size=32)
    test_input = encode("hello")
    target = char_to_idx[' ']

    print("Before training:")
    probs_before = forward_pass(test_input, weights)
    print(f"Probability of ' ': {probs_before[target]:.4f}")

    # Train on this example 100 times
    for _ in range(100):
        backward_pass(test_input, target, weights, learning_rate=0.1)

    print("\nAfter 100 updates:")
    probs_after = forward_pass(test_input, weights)
    print(f"Probability of ' ': {probs_after[target]:.4f}")