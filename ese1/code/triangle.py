a = float(input())
b = float(input())
c = float(input())

if a <= 0 or b <= 0 or c <= 0:
    print("They are not edges!")
elif ((a**2 + b**2 == c**2) or
      (b**2 + c**2 == a**2) or
      (c**2 + a**2 == b**2)):
    print("Yes")
else:
    print("no")
