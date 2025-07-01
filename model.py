from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from numpy import argmax
from nltk.translate.bleu_score import corpus_bleu
from data import word_for_id
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers


def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, mask_zero=True))
	model.add(Dropout(0.3))
	model.add(LSTM(n_units, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.001)))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.001)))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

def transltate_sentence(model, tokenizer, sentence):
	# encode the source sentence
	source = tokenizer.texts_to_sequences([sentence])
	source = pad_sequences(source, maxlen=model.input_shape[1], padding='post')
	# predict the translation
	translation = predict_sequence(model, tokenizer, source)
	return translation

# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append([raw_target.split()])
		predicted.append(translation.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))