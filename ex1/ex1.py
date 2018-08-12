

#def createMatrix():




matrix = [[-1 for y in xrange(5)] for x in xrange(2)] #defines a 2x5 matrix 
#defining A 
a = [[-1, -1,  1, -1,  1],
     [-1,  1, -1,  1, -1],
     [-1,  1,  1,  1, -1],
     [ 1, -1, -1, -1,  1],
     [ 1, -1, -1, -1,  1]]
#defining A inverted
b =[[ 1, -1, -1, -1,  1],
	[ 1, -1, -1, -1,  1],
	[-1,  1,  1,  1, -1],
	[-1,  1, -1,  1, -1],
	[-1, -1,  1, -1, -1]]




#defining inverted A

for x in range(0, 5):
    for y in range(0, 5):
    	print '{:>3}'.format(b[x][y]),
    print




#createMatrix()