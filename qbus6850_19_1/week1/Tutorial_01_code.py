#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 14:01:50 2017

@author: steve
"""

"""
Hello World!
"""

print("Hello World!")

"""
Lists
"""

my_list = [1, 2, 4, 8, 16]

print(my_list)

# Index 0 is the first item
print(my_list[0])

# Adding items to the list
my_list.append(32)

print(len(my_list))

my_new_list = list()

my_new_list.append(10)
my_new_list.append(12)

print(my_new_list)

"""
User Input
"""

# Capture the user name
name = input("Enter your name:")

print("Welcome " + name)

"""
Conditionals
Decision making
"""

input_number = int(input("Enter a number:"))

if input_number > 3:
    print("number is greater than 3")
elif input_number == 3:
    print("number is equal to 3")
else:
    print("number is less than 3")

"""
Loops
Be careful not to create an infinite loop!
"""

for number in my_list:
    print(number)

for i in range(len(my_list)):
    print(my_list[i])

count = 0
while (count < len(my_list)):
    print(my_list[count])
    count = count + 1

"""
Functions
print is also a function
"""

def add(a, b):
    return a + b

c = add(10, 5)
print(c)


"""
String Formatting
"""

year = int(input("Enter your birthh year: "))

age = 2017 - year

fancy_string = "Hi {0}, you are {1} years old".format(name, age)

print(fancy_string)

"""
Classes and Objects
"""

class Customer(object):
    #A bank customer
    
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance
    
    def __str__(self):
        return "Account for {0}, balance of {1}".format(self.name, self.balance)
    
    def widthdraw(self, amount):
        self.balance = self.balance - amount
        return self.balance
    
    def deposit(self, amount):
        self.balance = self.balance + amount
        return self.balance
    
    

new_customer = Customer("Steve", 1000.00)

print(new_customer.balance)
print(new_customer.name)

print(new_customer)

new_customer.widthdraw(100)

print(new_customer.balance)