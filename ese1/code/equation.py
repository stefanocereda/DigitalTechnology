a = input("Give me a number: ")
b = input("Give me a number: ")
c = input("Give me a number: ")

a = float(a)
b = float(b)
c = float(c)

delta = b ** 2 - 4*a*c
x1 = ((-b + delta ** 1/2)) / (2*a)
x2 = (-b - delta ** 1/2) / (2*a)

print("The solutions of {}x^2 + {}x + {} = 0 are {} and {}".format(a,b,c,x1,x2))
