import codecs
import numpy as np

class DataHandler():
	def __init__(self, file="goblet_book.txt"):
		self.filename = file
		self.text = codecs.open(self.filename, encoding="utf8")
		self.data = self.text.read()
		temp = set(self.data)
		self.len = len(temp)
		self.text.close()
		self.letterToIndex = {}
		self.indexToLetter = []
		i = 0
		for letter in temp:
			self.indexToLetter.append(letter)
			self.letterToIndex[letter] = i
			i += 1
		


	def getInputOutput(self, start, size=25):
		x = self.data[start:start+size]
		y = self.data[start+1:start+size+1]
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


if __name__ == "__main__":
	datahandler = DataHandler()
	for i in range(10):
		print(datahandler.getIndex("h"))