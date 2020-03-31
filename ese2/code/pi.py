max_iters = int(input("How many iterations? "))
pi = 0
for n_iter in range(0, max_iters):
    if n_iter % 2 == 0:
        pi = pi + 1 / (2*n_iter +1)
    else:
        pi = pi - 1 / (2*n_iter +1)

    if (n_iter + 1) % 10 == 0:
        print("The value of pi after {} iterations is {}".format(n_iter+1, 4*pi))

print("The final approximation is {}".format(4*pi))
