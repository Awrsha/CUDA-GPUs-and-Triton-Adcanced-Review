class Person:
    def __init__(self, age, name):
        self.age = age
        self.name = name

# Create person1
person1 = Person(25, "elliot")
print(f"age: {person1.age}\tname: {person1.name}")

# Create person2
person2 = Person(20, "not elliot")
print(f"age: {person2.age}\tname: {person2.name}")