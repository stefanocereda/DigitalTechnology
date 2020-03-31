# Step 1: read IDs
student_ids = set()
while True:
    id = input("Give me an id (-1 to end): ")
    if id == '-1':
        break
    student_ids.add(id)

# Step 2: read marks
student_marks = {}
for id in student_ids:
    mark = input("Give me the mark of student {} ".format(id))
    mark = int(mark)
    student_marks[id] = mark

# Step 3: students >= 18
passed = []
for id in student_ids: # key iterator
    if student_marks[id] >= 18:
        passed.append(id)
# This is equivalent:
# passed = [id for id in student_ids if student_marks[id] >= 18]
print("These students passed the exam:", passed)

# Step 4: min, max, avg
# We can access the values of a dictionary using the .values() method
marks = student_marks.values()
print("The minimum mark is", min(marks))
print("The maximum mark is", max(marks))
avg_mark = sum(marks) / len(marks)
print("The average mark is", avg_mark)

# Step 5: standard deviation
squared_errors = [(mark - avg_mark)**2 for mark in marks]
std = (sum(squared_errors) / len(marks)) ** (1/2)
print("The standard deviation is", std)
