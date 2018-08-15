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
	[-1, - 1,  1,  1, -1],
	[-1,  1, -1,  1, -1],
	[-1, -1,  1, 1, -1]]
train2 =[[ 1, -1, -1, -1,  1],
	[ 1, -1, -1, -1,  1],
	[ 1,  1,  -1,  1, -1],
	[-1,  -1, -1,  1, -1],
	[-1, -1,  1, -1, -1]]
train3 = [[1, -1,  1, -1,  1],
     [-1,  1, -1,  1, -1],
     [-1,  -1,  1,  -1, -1],
     [ 1, -1, -1, -1,  1],
     [ 1, -1, -1, 1,  1]]
train4 = [[-1, -1,  1, -1,  1],
     [-1,  1, -1,  1, -1],
     [-1,  1,  1,  1, -1],
     [ 1, -1, -1, -1,  1],
     [ 1, 1, 1, -1,  1]]
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
activateResult = 0
wmodified = 0

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
	global activateResult
	activateResult = activationFunc(output)
	if (label != activateResult):
		return 0
	else:
		return 1

def correctWeights(inputMatrix, label):
	global w
	global wmodified
	wmodified = 1
	for x in range(0, 5):
		for y in range(0, 5):
			w[x][y] = w[x][y] + eta*(label-activateResult)*inputMatrix[x][y]


def perceptromAlgorithm(inputMatrix, label):
	output = processInput(inputMatrix)
	goodLabel = isCorrectLabel(output, label)
	if(goodLabel != 1):
		correctWeights(inputMatrix, label)
		perceptromAlgorithm(inputMatrix, label)

def trainingCases(number):
	return{
		1:train1,
        2:train2,
        3:train3,
        4:train4,
        5:train5,
        6:train6,
	}[number]

def trainingLoop(actualTrainingCase):
	global wmodified, trainLabel
	for i in range (1,7):
		wmodified = 0
		if(actualTrainingCase == i):
			continue
		perceptromAlgorithm(trainingCases(i), trainLabel[i-1])
		if (wmodified == 1):
			actualTrainingCase = i
			trainingLoop(actualTrainingCase)
			return

def testCase(inputMatrix):
	output = processInput(inputMatrix)
	if(activationFunc(output) == -1):
		print "The character is A"
	else:
		print "The character is inverted A"

# main program # 
trainingLoop(0)
print "Weigth matrix"
printMatrix(w)

print
testCase(test1)
print "It should be inverted A"
print
testCase(test2)
print "It should be inverted A"
print

testCase(test3)
print "It should be inverted A"
print

testCase(test4)
print "It should be A"
print

testCase(test5)
print "It should be A"
print

testCase(test6)
print "It should be A"



