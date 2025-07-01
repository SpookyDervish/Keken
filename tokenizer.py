from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer


def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

def encode_sequences(tokenizer: Tokenizer, length: int, lines: list[str]):
	"""Encode and pad sequences."""
	X = tokenizer.texts_to_sequences(lines)
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

def encode_output(sequences, vocab_size: int):
	"""One hot encode target sequence."""
	ylist = []
	for seq in sequences:
		encoded = to_categorical(seq, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y