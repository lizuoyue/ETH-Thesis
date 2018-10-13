a = [0,1,2,3,4]
n = 5
for i in range(n):
	a.pop(0)
	for j in range(n):
		a.append(n*(i+1)+j)

print(a)
