#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 18:04:26 2017

@author: steve
"""

import random

a = random.randint(1,10)

running = True

while running:
    
    uinput = input("Guess a number from 1-10 (type exit to end): ")

    if uinput != "exit":
        guess = int(uinput)
        if guess == a:
            print("You got it!")
            running = False
        else:
            print("Please try again!")
        
    else:
        running = False