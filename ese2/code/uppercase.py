phrase = input("Tell me something: ")

# let's find the first space so to have the index where the first word stops
idx = 0
while idx < len(phrase) and phrase[idx] != ' ':
    idx = idx + 1
# at this point, phrase[idx] = ' '
end_word_idx = idx-1

# Equivalent code
#for idx in range(len(phrase)):
#    if phrase[idx] == ' ':
#        break

upper_word = phrase[0:end_word_idx+1].upper()
print(upper_word)


# More pythonic solution: phrase.split(' ')[0].upper()
