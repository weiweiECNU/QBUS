#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 18:04:50 2017

@author: steve
"""

word = "chair"

n_unique_chars = len(set(word))

guess_state = ["_", "_", "_", "_", "_"]

guessed_letters = set()
guessed_correct = set()

n_guesses = 10

running = True

while running:
    uinput = input("Guess a letter: ")
    
    if uinput not in guessed_letters:
        
        guessed_letters.add(uinput)
        
        locs = [pos for pos, char in enumerate(word) if char == uinput]
        
        if len(locs) > 0:
            guessed_correct.add(uinput)
        
            for l in locs:
                guess_state[l] = uinput
        else:
            n_guesses = n_guesses - 1
        
        
        if len(guessed_correct) < n_unique_chars:
            
            
            if (n_guesses <= 0):
                running = False
                print("No more guesses left")
            else:
                print("{0} guesses left".format(n_guesses))
                print("Guessed letters: {0}".format(guessed_letters))
                print(guess_state)
        else:
            running = False
            print("You win!")
            print(guess_state)
            
    else:
        print("You have already guessed {0}. Ignoring.".format(uinput))
        print("Guessed letters: {0}".format(guessed_letters))
        print("{0} guesses left".format(n_guesses))
        print(guess_state)
    