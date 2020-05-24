arr = []
print('hello')
art1 = ['1', 0.1]
art2 = ['2', 0.7]
art3 = ['3', 0.2]

arts = [art1, art2, art3]

for art in arts:
    arr = arr+([art[0]]*(int)(100*art[1]))


print(arr)
print(len(arr))
