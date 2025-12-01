from tokenizer import encode, decode, char_to_idx, idx_to_char

def create_training_data(text, context_length=5):
    """
    Split text into input sequences and their next character.
    context_length: how many characters to look at to predict the next one
    """
    encoded = encode(text)

    # Store all input sequences (what the model sees).
    inputs = []

    # Store all target characters (what the model should predict).
    # If input is the question, target is the answer we want the model to produce.
    targets = []

    for i in range(len(encoded) - context_length):
        # [1:3] this is the slice operation.
        # It means "get from index 1 up to (but not including) index 3"
        # So encoded[i:i + context_length] gets a slice of length context_length
        inputs.append(encoded[i:i + context_length])

        targets.append(encoded[i + context_length])

    return inputs, targets

# Example usage:
if __name__ == "__main__":
    sample_text = "hello world"
    inputs, targets = create_training_data(sample_text, context_length=3)

    for inp, tgt in zip(inputs, targets):
        print(f"Input: {decode(inp)} -> Target: {idx_to_char[tgt]}")