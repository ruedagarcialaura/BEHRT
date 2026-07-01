import pickle

# Path to the file (change it if your file is inside a folder like 'data/')
vocab_path = 'data/vocab.pkl'

print(f"Loading file: {vocab_path}...\n")

try:
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        
    print("=========================================")
    print(f"Object type: {type(vocab)}")
    
    # 1. If the file is a standard Python dictionary
    if isinstance(vocab, dict):
        print(f"Total words in vocabulary: {len(vocab)}")
        print("\n--- First 10 elements ---")
        for i, (word, index) in enumerate(vocab.items()):
            print(f"{word} -> ID: {index}")
            if i >= 9:
                break
                
    # 2. If the file is a custom Class (Typical in the BEHRT library)
    elif hasattr(vocab, 'word2idx'):
        print(f"Total words in vocabulary: {len(vocab.word2idx)}")
        print("\n--- BEHRT Vocab class attributes ---")
        print(f"Has 'word2idx': {hasattr(vocab, 'word2idx')}")
        print(f"Has 'idx2word': {hasattr(vocab, 'idx2word')}")
        
        print("\n--- First 10 elements (word2idx) ---")
        for i, (word, index) in enumerate(vocab.word2idx.items()):
            print(f"{word} -> ID: {index}")
            if i >= 9:
                break
    
    # 3. If it is a different/unrecognized object
    else:
        print("\nThe object is a different class. Its methods/attributes are:")
        print(dir(vocab))

    print("\n=========================================")
    print("Inspection completed successfully!")

except FileNotFoundError:
    print(f"ERROR: File '{vocab_path}' not found. Make sure the path is correct.")
except Exception as e:
    print(f"An error occurred while trying to read the file: {e}")