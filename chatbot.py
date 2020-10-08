"""
Chatbot application
"""

# Import necessary modules
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import tflearn as tf
import tensorflow
import random


# Create training data and label
class data:
	def __init__(self,data):
		with open(data) as file:
			self.data = json.load(file)
		self.words = []
		self.labels = []
		self.docs_x = []
		self.docs_y = []
		self.X = []
		self.stemmer = LancasterStemmer()
		self.enc = OneHotEncoder(sparse=False)
		self.label = LabelEncoder()
 
	def training_data(self):
		for tag in self.data['data']:
			for pattern in tag['questions']:
				wrds = nltk.word_tokenize(pattern)
				self.words.extend(wrds)
				self.docs_x.append(wrds)
				self.docs_y.append(tag['tag'])

			if tag['tag'] not in self.labels:
				self.labels.append(tag['tag'])

		self.words = [self.stemmer.stem(w.lower()) for w in self.words if w != '?']
		self.words = sorted(list(set(self.words)))
		self.labels = sorted(self.labels)
		
		# Loop to create the training data
		for idx,doc in enumerate(self.docs_x):
			temp = [self.stemmer.stem(w) for w in doc if w != '?']
			self.sack = [0 for _ in range(len(self.words))]
			for t in temp:
				if t in self.words:
					self.sack[self.words.index(t)] = 1
				else:
					self.sack[self.words.index(t)] = 0
			self.X.append(self.sack)
		
		self.output = self.label.fit_transform(self.docs_y)
		# Training set
		self.X = np.array(self.X)
		# Labels
		self.Y = self.enc.fit_transform(self.output.reshape(-1,1))

# Class to train the network
class network:
	def __init__(self,obj,train):
		tensorflow.reset_default_graph()
		self.obj = obj
		self.train = train
		self.forward()

	# Network Architecture
	def forward(self):
		self.net = tf.input_data(shape=[None,len(self.obj.X[0])])
		self.net = tf.fully_connected(self.net,16,activation='relu')
		self.net = tf.fully_connected(self.net,8,activation='relu')
		self.net = tf.fully_connected(self.net,len(self.obj.Y[0]),activation='softmax')
		self.net = tf.regression(self.net,optimizer='adam',loss='categorical_crossentropy')
		self.model = tf.DNN(self.net)
	
	# Run or load the model 
	def run(self):
		if self.train:
			self.model.fit(self.obj.X, self.obj.Y, n_epoch = 200, batch_size=8,show_metric=True)
			self.model.save("model.chatbotmodel")
			print("Model saved to disk")
		else:
			self.model.load("model.chatbotmodel")
			print("Model successfully loaded")
			self.chat()
			
	# Preprocess the data for prediction
	def preprocess(self,sentence,words):
		self.sack = [0 for _ in range(len(words))]
		self.sentence = nltk.word_tokenize(sentence)
		self.sentence = [self.obj.stemmer.stem(wrds.lower()) for wrds in self.sentence]

		for idx,i in enumerate(self.sentence):
			if i in words:
				self.sack[self.obj.words.index(i)] = 1

		self.sack = np.array(self.sack)
		return self.sack

	def chat(self):
		print("How may I help you!!!!(press quit to exit)")
		while True:
			inp = input("You : ")
			if inp == "quit":
				break
			# Predict the model's output
			predict = self.model.predict([self.preprocess(inp,self.obj.words)])
			result = self.obj.labels[np.argmax(predict)]
			for tag in self.obj.data['data']:
				if tag['tag'] == result:
					print(random.choice(tag['responses']))


def main():
	ct = data('Data/intents.json')
	ct.training_data()
	net = network(ct,False)
	net.run()
	
if __name__ == "__main__":
	main()