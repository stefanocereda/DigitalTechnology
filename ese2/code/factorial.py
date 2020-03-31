number = int(input("Give me a number: "))

result = 1
for factor in range(1, number+1):
    result = result * factor
print("{}! = {}".format(number, result))







factor = 1
result = 1
while factor <= number:
    result = result * factor
    factor = factor + 1

