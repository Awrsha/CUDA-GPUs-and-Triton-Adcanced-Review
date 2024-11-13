struct Person:
    var age: Int
    var name: String

fn main():
    # Create person1
    let person1 = Person(age=25, name="elliot")
    print(f"age: {person1.age}\tname: {person1.name}")

    # Create person2
    let person2 = Person(age=20, name="not elliot")
    print(f"age: {person2.age}\tname: {person2.name}")