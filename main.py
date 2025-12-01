from model import initialize_weights
from train import generate_text, train
from data import encode, idx_to_char
from model import forward_pass

if __name__ == "__main__":
  training_text = "the cat sat on the mat the dog sat on the log the rat sat on the hat"
  weights = initialize_weights(vocab_size=27, embed_size=32)

  # Bulk train
  train(training_text, weights, epochs=50, learning_rate=0.1, context_length=5)
  # Fine-tune with a smaller learning rate
  train(training_text, weights, epochs=100, learning_rate=0.025, context_length=5)
  # Final fine-tune with an even smaller learning rate
  train(training_text, weights, epochs=250, learning_rate=0.01, context_length=5)

  # Test it
  test_input = encode("on th")
  probs = forward_pass(test_input, weights)
  predicted_idx = probs.index(max(probs))
  print(f"\nTest: 'on th' -> predicted next character: '{idx_to_char[predicted_idx]}'")

  # Generate text using our trained model
  generate_text("the c", weights, length=25, context_length=5)
  generate_text("on th", weights, length=25, context_length=5)