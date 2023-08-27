{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPSC+mn3OLaDRjzDylx3TQz",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Ihsaniya/MyCap_Python_Assignments/blob/main/Ass_2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def fibonacci(n):\n",
        "  fib_seq=[0,1]\n",
        "  for i in range(2,n):\n",
        "    next_fib = fib_seq[i-1]+fib_seq[i-2]\n",
        "    fib_seq.append(next_fib)\n",
        "  return fib_seq\n",
        "n=int(input(\"Enter the fibonacci number\"))\n",
        "fib_num = fibonacci(n)\n",
        "print(\"Fibonacci Sequence:\",fib_num)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "R0X7_toCwZV7",
        "outputId": "32127e19-e41a-4770-f3a8-3b5548ba8663"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Enter the fibonacci number5\n",
            "Fibonacci Sequence: [0, 1, 1, 2, 3]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "O1iWkCJCPgbA"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}