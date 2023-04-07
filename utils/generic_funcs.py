import re
import os
import numpy as np


def search_between(start, end, string):
    """Searches for a string between two other strings

    Args:
        start (str): begining string, must be before end, the string between start and end will be returned
        end (str): ending string, must be after start, the string between start and end will be returned
        string (str): string to search for

    Returns:
        str: string between start and end if match found, else None
    """
    return re.search('%s(.*)%s' % (start, end), string).group(1)


def progress_bar(current, total, bar_length=20, message='Loading'):
    """Prints a progress bar to the console

    Args:
        current (float): current value
        total (float): total value
        bar_length (int, optional): length of bar. Defaults to 20.
        message (str, optional): loading message. Defaults to 'Loading'.
    """
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(message + f': [{arrow}{padding}] {int(fraction * 100)}%', end=ending)
    
def selectyesno(prompt):
    '''
    given a prompt with a yes / no input answer, return the boolean value of the given answer
    :param prompt: str a prompy with a yes / no answer
    :return: bool truth value of the given answer: yes -> True, no -> False
    '''
    print(prompt)  # print the user defined yes / no question prompt
    # list of understood yes inputs, and a list of understood no inputs
    yes_choices, no_choices = ['yes', 'ye', 'ya', 'y', 'yay'], ['no', 'na', 'n', 'nay']
    # use assignment expression to ask for inputs until an understood input is given
    while (choice := input('enter: (y / n) ').lower()) not in yes_choices + no_choices:
        print('input not understood: {} '.format(choice))
    # if the understood input is a no, it returns false, if it is a yes, it returns true
    return choice in yes_choices

