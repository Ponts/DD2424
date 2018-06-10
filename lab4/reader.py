import codecs
import numpy as np
import json
from pprint import pprint
import re

class DataHandler():
	def __init__(self, file="goblet_book.txt", mode="Potter"):
		if mode == "Potter":
			self.text = codecs.open(file, "r", encoding="utf8")
			self.data = self.text.read()
		else:
			self.EOT = "Ö"
			for file in ["master_2018.json", "master_2017.json", "master_2016.json", "master_2015.json"]:
				self.text = codecs.open(file)
				self.data = ""
				self.tweets = json.load(self.text)

				for i in range(len(self.tweets)):
					try:
						line = re.sub(r'[^\x00-\x7F]+',' ',self.tweets[i]["full_text"]) 
						line = re.sub(r'https?:\/\/.*[\r\n]*', '', line) + self.EOT
						#if self.EOT in self.tweets[i]["full_text"]:
						#	print("Tweet has EOF")
					except KeyError:
						line = re.sub(r'[^\x00-\x7F]+',' ',self.tweets[i]["text"]) 
						line = re.sub(r'https?:\/\/.*[\r\n]*', '', line) + self.EOT
						#if self.EOT in self.tweets[i]["text"]:
						#	print("Tweet has EOF")
					self.data+=line



		temp = set(self.data)
		self.len = len(temp)
		self.text.close()
		self.letterToIndex = {}
		self.indexToLetter = {}
		i = 0
		for letter in temp:
			self.indexToLetter[i] = letter
			self.letterToIndex[letter] = i
			i += 1
		if mode != "Potter":
			self.endChar = self.getIndex(self.EOT)
		print("Found " + str(self.len) + " distinct characters")
		self.text.close()


	def getInputOutput(self, start, size=25):
		x = self.data[start:start+size]
		y = self.data[start+1:start+1+size]
		encodedX = np.zeros((self.len,size))
		encodedY = np.zeros((self.len,size))
		for i in range(len(x)):
			encodedX[:,i] = self.getOneHot(self.getIndex(x[i]))
			encodedY[:,i] = self.getOneHot(self.getIndex(y[i]))
		return encodedX, encodedY
		

	def getIndex(self, letter):
		return self.letterToIndex[letter]

	def getLetter(self, index):
		return self.indexToLetter[index]

	def getOneHot(self, index):
		x = np.zeros((self.len))
		x[index] = 1
		return x


	def encodedToLetter(self, encoded):
		index = np.argmax(encoded)
		return self.getLetter(index)

	def toString(self, encoded):
		string = ""
		for t in range(encoded.shape[1]):
			string += self.encodedToLetter(encoded[:,t])
		return string


if __name__ == "__main__":
	datahandler = DataHandler(mode="Trump")
	