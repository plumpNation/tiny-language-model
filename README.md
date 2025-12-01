# Tiny Language Model - Learning Guide

A simple language model built from scratch with **zero libraries** (except basic Python math). Perfect for understanding how AI language models actually work under the hood!

## What Does This Do?

This model learns patterns in text and can predict what character comes next. After training on simple phrases like "the cat sat on the mat", it learns to complete patterns:

- Input: "the c" → Predicts: "a" (because it learned "cat" comes after "the c")
- Input: "sat o" → Predicts: "n" (because it learned "on" comes after "sat")

It's like autocomplete on your phone, but tiny and simple enough to understand completely.

---

## How It Works: Step by Step

### Step 1: Converting Text to Numbers (Tokenization)

**Why?** Computers can't work with letters directly - they need numbers.

```python
vocab = " abcdefghijklmnopqrstuvwxyz"
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
```

This creates a dictionary where:
- Space = 0
- 'a' = 1
- 'b' = 2
- ... and so on

**Example:**
- "hello" becomes [8, 5, 12, 12, 15]
- We can convert it back: [8, 5, 12, 12, 15] → "hello"

**Functions:**
- `encode()` - Converts text to numbers
- `decode()` - Converts numbers back to text

---

### Step 2: Preparing Training Data

**Why?** The model needs to see examples of "given these characters, predict the next one."

```python
create_training_data("the cat", context_length=5)
```

This splits text into learning examples:
- Input: "the c" → Target: "a"
- Input: "he ca" → Target: "t"
- Input: "e cat" → Target: " " (space)

The model will learn from thousands of these examples.

**Key term:** `context_length` = how many characters the model looks at to predict the next one (we use 5).

---

### Step 3: Initializing the Model (Creating Random Starting Weights)

**Why random?** The model has to start somewhere! It begins with random guesses and gradually learns the right patterns.

```python
weights = initialize_weights(vocab_size=27, embed_size=32)
```

This creates two main components:

#### 3a. Embeddings (Character Representations)
Each character gets converted into 32 numbers. At first, these are random, but during training they become meaningful.

- Character 'a' → [0.05, -0.12, 0.08, ..., 0.03] (32 random numbers)
- Character 'b' → [-0.02, 0.15, -0.07, ..., 0.11] (32 different random numbers)

**Why 32 numbers per character?** One number can't capture everything about a character. Multiple numbers let the model learn complex patterns like "is this a vowel?", "does this start words?", "is this common?". The model figures out what each number means during training.[^1]

#### 3b. Output Weights (W_output)
A grid of numbers (32 × 27) that transforms the model's understanding back into character predictions.

**Total parameters:** About 4,800 numbers that will be adjusted during training.

---

### Step 4: Forward Pass (Making Predictions)

**What is it?** Taking input text and running it through the model to get a prediction.

```python
probabilities = forward_pass("hello", weights)
```

**The process:**

1. **Look up embeddings:** Convert each input character to its 32 numbers
   - 'h' → [0.05, -0.12, ...]
   - 'e' → [-0.02, 0.15, ...]
   - 'l' → [0.08, 0.21, ...]
   - etc.

2. **Create context:** Average all the embeddings to get one representation of the input
   - Result: One list of 32 numbers representing "hello"

3. **Calculate output scores:** Multiply context by W_output weights
   - Result: 27 numbers (one score for each possible next character)

4. **Convert to probabilities:** Use softmax[^2] to turn scores into percentages
   - Result: [0.02, 0.15, 0.03, ...] (probabilities that sum to 1.0)
   - This means: 2% chance next char is space, 15% chance it's 'a', 3% chance it's 'b', etc.

**Key functions:**
- `forward_pass()` - Runs the whole prediction process
- `softmax()` - Converts scores to probabilities[^2]

---

### Step 5: Calculating Loss (How Wrong Were We?)

**What is it?** A number that measures how bad our prediction was.

```python
loss = cross_entropy_loss(probabilities, correct_answer)
```

**Example:**
- Model predicted: 70% chance next character is 'a'
- Actual answer: 'a'
- Loss: 0.36 (pretty good!)

vs.

- Model predicted: 5% chance next character is 'a'
- Actual answer: 'a'
- Loss: 3.00 (really bad!)

**The formula:** `-log(probability_of_correct_answer)`[^3]

**Why this formula?** It heavily punishes confident wrong predictions. If you're 99% sure of the wrong answer, you get a huge loss. If you're uncertain (50/50), the loss is moderate.

**Goal of training:** Make the loss as small as possible!

---

### Step 6: Backward Pass (Learning from Mistakes)

**What is it?** Adjusting all the weights to make better predictions next time.

```python
backward_pass(input_sequence, correct_answer, weights, learning_rate=0.05)
```

**The process:**

1. **Calculate error:** How different was our prediction from the correct answer?
   - Predicted probabilities: [0.02, 0.15, 0.03, 0.70, ...]
   - Correct answer was index 3
   - Error: [0.02, 0.15, 0.03, -0.30, ...] (subtract 1 from correct position)

2. **Update output weights:** Adjust W_output based on the error
   - If we predicted too much 'b' and not enough 'a', increase weights that favor 'a'

3. **Update embeddings:** Adjust the character representations
   - If 'h' contributed to a wrong prediction, adjust its 32 numbers slightly

4. **Learning rate:** Controls how big the adjustments are
   - 0.05 = small, careful steps (stable but slow)
   - 0.5 = big steps (fast but might overshoot)

**Key insight:** We use calculus (gradients) to figure out which direction to adjust each weight. This is called "gradient descent"[^4] - walking downhill toward lower loss.

---

### Step 7: Training Loop (Putting It All Together)

**What is it?** Repeating the forward pass and backward pass many times until the model learns.

```python
train(training_text, weights, epochs=100, learning_rate=0.05)
```

**One epoch = one complete pass through all training examples**

**What happens each epoch:**

1. Loop through every training example:
   - Input: "the c" → Target: "a"

2. For each example:
   - **Forward pass:** Make a prediction
   - **Calculate loss:** See how wrong we were
   - **Backward pass:** Adjust weights to do better

3. Print average loss every 10 epochs to track progress

**What you'll see:**
```
Epoch 10/100, Average Loss: 2.8543
Epoch 20/100, Average Loss: 2.1234
Epoch 30/100, Average Loss: 1.6789
...
Epoch 100/100, Average Loss: 0.4521
```

**Loss going down = model is learning!**

After 100 epochs, those random weights have become meaningful patterns.

---

### Step 8: Generating Text (Using the Trained Model)

**What is it?** Using our trained model to predict and generate new text character by character.

```python
generate_text("the c", weights, length=20)
```

**The process:**

1. Start with seed text: "the c"

2. Predict next character:
   - Run forward pass on "the c"
   - Get probabilities: [0.01, 0.85, 0.02, ...]
   - Pick highest probability → 'a'
   - New text: "the ca"

3. Predict again:
   - Run forward pass on "he ca" (last 5 characters)
   - Pick highest probability → 't'
   - New text: "the cat"

4. Keep going for 20 characters

**Result:** "the cat sat on the m"

The model learned patterns from training data and can now generate similar text!

---

## Running the Code

1. Save the code as `tiny_language_model.py`

2. Run it:
```bash
python tiny_language_model.py
```

3. Watch it train and generate text!

---

## Key Concepts Explained

### What Are Parameters?
All the numbers the model learns (embeddings + weights). Our model has ~4,800 parameters. GPT-3 has 175 billion!

### What Is Training?
Showing the model examples and adjusting its parameters to make better predictions. Like practicing basketball - you miss shots at first, but gradually improve.

### Why Does It Work?
After seeing "the cat" together many times, the model's parameters encode this pattern. When it sees "the c", the math naturally points toward 'a' as the next character.

### What Makes It "Intelligent"?
It's not truly intelligent - it's pattern recognition. But when patterns get complex enough (in bigger models with more data), it can seem intelligent!

---

## Experiment Ideas

1. **Change the training text** - Try song lyrics, poems, or code
2. **Adjust embed_size** - Try 16 or 64 instead of 32
3. **Change context_length** - Try looking at 3 or 10 previous characters
4. **Modify learning_rate** - See how training speed changes
5. **Add more training data** - Does it learn better patterns?

---

## Limitations

This is a **toy model** for learning. Real language models:
- Have billions of parameters (we have ~4,800)
- Use GPUs for speed (we use CPU)
- Train on gigabytes of text (we use one sentence)
- Use sophisticated techniques like attention, transformers, etc.

But the core ideas are the same!

---

## What's Next?

To learn more about language models:

1. **Neural Networks Basics** - 3Blue1Brown's video series
2. **Transformers** - The architecture used in ChatGPT
3. **Attention Mechanism** - How models focus on relevant parts
4. **PyTorch/TensorFlow** - Libraries that make building real models practical

You've built a language model from scratch - that's huge! You understand the fundamentals. Everything else is just adding layers of sophistication on top of these core concepts.

---

## Footnotes

[^1]: **Embeddings:** Think of embeddings like coordinates on a map. Each character gets a location in 32-dimensional space. Characters that behave similarly end up close together. After training, 'c' and 'k' might be near each other (both consonants), while 'a' and 'e' are near each other (both vowels). The model learns these positions automatically. [Learn more about embeddings](https://jalammar.github.io/illustrated-word2vec/)

[^2]: **Softmax Function:** Converts any numbers into probabilities (0 to 1) that sum to 1. Formula: For each number, do e^number, then divide by the sum of all e^numbers. The exponential (e^x) makes bigger numbers much bigger and smaller numbers much smaller, which helps the model be more confident. [Learn more about softmax](https://www.youtube.com/watch?v=ytbYRIN0N4g)

[^3]: **Cross-Entropy Loss:** Measures how different two probability distributions are. Formula: -log(probability_assigned_to_correct_answer). The log function has a special property: it grows really fast as probabilities get close to 0. So if you give 1% probability to the right answer, your loss is HUGE. This encourages the model to be confident when it's right. [Learn more about cross-entropy](https://www.youtube.com/watch?v=ErfnhcEV1O8)

[^4]: **Gradient Descent:** Imagine you're blindfolded on a hilly landscape and trying to reach the lowest point. You feel which direction is downhill and take a step that way. Repeat. That's gradient descent! In our case, the "landscape" is all possible weight values, and the "height" is the loss. We use calculus to find which direction is "downhill" (reduces loss) and adjust weights that way. [Learn more about gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w)

---

## Questions?

If something doesn't make sense, try:
1. Running the code and printing intermediate values
2. Changing one thing at a time to see what happens
3. Drawing diagrams of how data flows through the model

The best way to learn is to experiment!

**Remember:** Every expert started exactly where you are now. Keep exploring! 🚀