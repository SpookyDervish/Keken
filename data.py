import re, string, sys
from numpy import array
from unicodedata import normalize
from pickle import dump, load


def load_doc(file_path: str) -> str:
    """Read the text from a file in the `rt` mode."""
    with open(file_path, "rt", encoding="utf-8") as file:
        return file.read()
    
def to_pairs(doc: str) -> list[str]:
    """Split a loaded document into sentences."""
    lines = doc.strip().split("\n")
    pairs = [line.split("\t") for line in lines]
    return pairs

def clean_pairs(lines: list[str]) -> list[str]:
    """Remove empty pairs and pairs with a single word."""
    print("Cleaning data pairs...")
    cleaned = []

    #re_printable = re.compile("[^%s]" % re.escape(string.printable))
    table = str.maketrans("", "", string.punctuation)

    for pair in lines:
        clean_pair = []
        for line in pair:
            #line = normalize("NFD", line).encode("ascii", "ignore").decode("utf-8")
            line = line.split()
            line = [word.lower() for word in line]
            line = [word.translate(table) for word in line]
            #line = [re_printable.sub("", word) for word in line]
            #line = [word for word in line if word.isalpha()]
            clean_pair.append(" ".join(line))
        cleaned.append(clean_pair)
    try:
        return array(cleaned)
    except ValueError:
        print("Your data is not structured correctly. Ensure each pair has a tab inbetween, and there are no empty lines.")
        sys.exit(1)

def save_clean_data(lines: list[str], file_path: str) -> None:
    with open(file_path, "wb") as f:
        dump(lines, f)
    print(f"Saved: {file_path}")

def load_clean_sentences(file_path: str):
    print(f"Loading: {file_path}")
    with open(file_path, "rb") as f:
        return load(f)
    print(f"Loaded: {file_path}")

def max_length(lines: list[str]) -> int:
    """Calculate the maximum length of sentences in a list."""
    return max(len(line.split()) for line in lines)

# map an integer to a word
def word_for_id(integer: int, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None