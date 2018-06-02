import codecs
import numpy as np

class DataHandler():
	def __init__(self, file="goblet_book.txt"):
		self.filename = file
		self.text = codecs.open(self.filename, encoding="utf8")
		temp = set(self.text.read())
		self.len = len(temp)
		self.text.close()
		self.text = codecs.open(self.filename, encoding="utf8")
		self.letterToIndex = {}
		self.indexToLetter = []
		i = 0
		for letter in temp:
			self.indexToLetter.append(letter)
			self.letterToIndex[letter] = i
			i += 1

		self.encoded = []
		for letter in self.text.read():
			self.encoded.append(self.getOneHot(self.getIndex(letter)))
		self.encoded = np.asarray(self.encoded).reshape(-1,self.len)

	def getInputOutput(self, start, size=25):
		return self.encoded[start:start+25], self.encoded[start+1:start+size+1]

	def getIndex(self, letter):
		return self.letterToIndex[letter]

	def getLetter(self, index):
		return self.indexToLetter[index]

	def getOneHot(self, index):
		x = np.zeros(self.len)
		x[index] = 1
		return x.reshape(1,-1)

	def getEncodedData(self):
		return self.encoded

	def encodedToLetter(self, encoded):
		index = np.argmax(encoded)
		return self.getLetter(index)


if __name__ == "__main__":
	datahandler = DataHandler()
