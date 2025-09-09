from datasets import load_dataset

def get_dataset():
    # Load the Hugging Face "emotion" dataset
    dataset = load_dataset("emotion")
    return dataset

if __name__ == "__main__":
    dataset = get_dataset()
    print(dataset)
