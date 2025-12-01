# Our tiny vocabulary - just lowercase letters and space
vocab = " abcdefghijklmnopqrstuvwxyz"

# Create mappings from characters to indices and vice versa
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}

def encode(text):
    """Convert text to list of numbers"""
    return [char_to_idx[ch] for ch in text.lower() if ch in char_to_idx]

def decode(indices):
    """Convert numbers back to text"""
    return ''.join([idx_to_char[i] for i in indices])

# Example usage:
if __name__ == "__main__":
    sample_text = "hello world"
    encoded = encode(sample_text)
    print("Encoded:", encoded)
    decoded = decode(encoded)
    print("Decoded:", decoded)