# -*- coding: utf-8 -*-
"""Ass_2.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/112XWp_rMkzHZs1lfdxaSpTBkfoDe_fKd
"""

def fibonacci(n):
  fib_seq=[0,1]
  for i in range(2,n):
    next_fib = fib_seq[i-1]+fib_seq[i-2]
    fib_seq.append(next_fib)
  return fib_seq
n=int(input("Enter the fibonacci number"))
fib_num = fibonacci(n)
print("Fibonacci Sequence:",fib_num)

