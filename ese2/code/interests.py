amount = 5000
interest = 5/100

# The \t character is a tabulation character
print("Year\tAmount")
for year in range(10):
    amount = amount * (1+interest)
    print("{}\t{:.2f}".format(year+1, amount))
#{:.2f} means write a float number in fixed-point notation with 2 decimal digits
# You can find other formatting options here:
#https://docs.python.org/3/library/string.html#format-string-syntax
