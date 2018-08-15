#defining A 
a = [[-1, -1,  1, -1,  1],
     [-1,  1, -1,  1, -1],
     [-1,  1,  1,  1, -1],
     [ 1, -1, -1, -1,  1],
     [ 1, -1, -1, -1,  1]]
#defining A inverted
ai =[[ 1, -1, -1, -1,  1],
	[ 1, -1, -1, -1,  1],
	[-1,  1,  1,  1, -1],
	[-1,  1, -1,  1, -1],
	[-1, -1,  1, -1, -1]]

#train cases
train1 =[[ 1, -1, -1, -1,  1],
	[ 1, -1, -1, 1,  1],
	[-1,  1,  1,  1, -1],
	[-1,  1, -1,  1, -1],
	[-1, -1,  1, 1, -1]]
train2 =[[ 1, -1, -1, -1,  1],
	[ 1, -1, -1, -1,  1],
	[ 1,  1,  1,  1, -1],
	[-1,  -1, -1,  1, -1],
	[-1, -1,  1, -1, -1]]
train3 = [[1, -1,  1, -1,  1],
     [-1,  1, -1,  1, -1],
     [-1,  -1,  1,  1, -1],
     [ 1, -1, -1, -1,  1],
     [ 1, -1, -1, -1,  1]]
train4 = [[-1, -1,  1, -1,  1],
     [-1,  1, 1,  1, -1],
     [-1,  1,  1,  1, -1],
     [ 1, -1, -1, -1,  1],
     [ 1, -1, 1, -1,  1]]
train5 = a
train6 = ai
trainLabel = [1, 1, -1, -1, -1, 1]

#test cases 
test1 =[[ 1, -1, -1, -1,  1],
	[ 1, 1, -1, 1,  1],
	[-1,  1,  1,  1, -1],
	[-1,  1, -1,  1, -1],
	[-1, -1,  1, -1, -1]]
test2 =[[ 1, -1, -1, -1,  1],
	[ 1, -1, -1, -1,  1],
	[-1,  -1,  -1,  1, -1],
	[-1,  1, -1,  1, -1],
	[-1, -1,  1, -1, -1]]
test3 =[[ 1, -1, -1, -1,  1],
	[ 1, -1, -1, -1,  1],
	[-1,  -1,  1,  1, -1],
	[-1,  -1, -1,  1, -1],
	[-1,  1,  1, -1, -1]]
test4 = [[-1, -1,  1, -1,  1],
     [-1,  1, 1,  -1, 1],
     [-1,  1,  1,  1, -1],
     [ 1, -1, -1, -1,  1],
     [ 1, -1, -1, -1,  1]]
test5 = [[-1, 1,  1, -1,  1],
     [-1,  1, -1,  -1, -1],
     [-1,  1,  1,  1, 1],
     [ 1, -1, -1, -1,  1],
     [ 1, -1, -1, -1,  1]]
test6 = [[-1, 1,  1, -1,  1],
     [1,  -1, -1,  1, -1],
     [-1,  1,  1,  1, -1],
     [ 1, -1, -1, -1,  1],
     [ 1, -1, -1, -1,  1]]
testLabel = [1, 1, 1, -1, -1, -1]

#weigths
w = [[0, 0, 0, 0, 0],
	 [0, 0, 0, 0, 0],
	 [0, 0, 0, 0, 0],
	 [0, 0, 0, 0, 0],
	 [0, 0, 0, 0, 0]]

eta = 0.5
activate = 0
counter = 0


def activationFunc(value):
	if (value >=0):
		return 1
	else:
		return -1

def printMatrix(matrix):
	for x in range(0, 5):
		for y in range(0, 5):
			print '{:>3}'.format(matrix[x][y]),
		print

def processInput(inputMatrix):
	output = 0
	for x in range(0, 5):
		for y in range(0, 5):
			output = output + w[x][y]*inputMatrix[x][y]
	return output
	
def isCorrectLabel(output, label):
	global activate
	activate = activationFunc(output)
	if (label != activate):
		return 0
	else:
		return 1

def correctWeights(inputMatrix, label):
	global w
	for x in range(0, 5):
		for y in range(0, 5):
			w[x][y] = w[x][y] - eta*(activate-label)*inputMatrix[x][y]

def perceptromAlgorithm(inputMatrix, label):
	output = processInput(inputMatrix)
	goodLabel = isCorrectLabel(output, label)
	global counter
	if(goodLabel != 1):
		correctWeights(inputMatrix, label)
		perceptromAlgorithm(inputMatrix, label)

	else:
		print output, activate


# main program # 
perceptromAlgorithm(a, trainLabel[4])
printMatrix(w)





