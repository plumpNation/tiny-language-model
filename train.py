from model import initialize_weights
from data import encode, create_training_data, idx_to_char
from model import forward_pass, backward_pass, cross_entropy_loss

def train(text, weights, epochs=100, learning_rate=0.1, context_length=5):
    """
    Train the model on text data.\n
    text: the training text\n
    weights: the model weights to be updated\n
    epochs: how many times to go through all the training data\n
    learning_rate: how big of a step to take when updating weights\n
    context_length: how many characters to look at to predict the next one
    """
    # Prepare training data
    input_sequences, next_characters = create_training_data(text, context_length)

    print(f"Training on {len(input_sequences)} examples for {epochs} epochs\n")

    for epoch in range(epochs):
        total_loss = 0

        # Train on each example
        for i in range(len(input_sequences)):
            probs = backward_pass(input_sequences[i], next_characters[i], weights, learning_rate)
            loss = cross_entropy_loss(probs, next_characters[i])
            total_loss += loss

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(input_sequences)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    print("\nTraining complete!")

def generate_text(seed_text, weights, length=20, context_length=5):
    """
    Generate new text starting from seed_text.
    seed_text: starting text (must be at least context_length characters)
    length: how many new characters to generate
    """
    # Start with the seed text
    generated = seed_text.lower()

    print(f"Starting with: '{seed_text}'")
    print(f"Generating {length} characters...\n")

    for _ in range(length):
        # Take last context_length characters as input
        context = generated[-context_length:]
        input_seq = encode(context)

        # Get prediction
        probs = forward_pass(input_seq, weights)

        # Pick the most likely next character
        predicted_idx = probs.index(max(probs))
        next_char = idx_to_char[predicted_idx]

        # Add to generated text
        generated += next_char

    print(f"Result: '{generated}'")
    print()
    print(" ------------------ ")
    print()

    return generated

# Example usage:
if __name__ == "__main__":
  training_text = "the cat sat on the mat the dog sat on the log the rat sat on the hat"
  weights = initialize_weights(vocab_size=27, embed_size=32)

  train(training_text, weights, epochs=50, learning_rate=0.1, context_length=5)

  # Test it
  test_input = encode("on th")
  probs = forward_pass(test_input, weights)
  predicted_idx = probs.index(max(probs))
  print(f"\nTest: 'on th' -> predicted next character: '{idx_to_char[predicted_idx]}'")

  # Generate text using our trained model
  generate_text("the c", weights, length=15, context_length=5)
  generate_text("on th", weights, length=15, context_length=5)