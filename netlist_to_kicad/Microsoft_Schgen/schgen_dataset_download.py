from datasets import load_dataset

# Load the SchGen dataset
dataset = load_dataset("microsoft/SchGen_dataset")
print(dataset['train'][0])