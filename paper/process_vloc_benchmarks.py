import re

# import pyperclip

while True:
    res = input("Some text")
    pattern = r"[-+]?(?:\d*\.*\d+)"
    # res = "53.2 / 85.8 / 95.3	3.5 / 11.7 / 25.2"
    matches = re.findall(pattern, res)
    numbers = list(map(float, matches))
    avg = sum(numbers) / len(matches)
    numbers.append(round(avg, 1))
    numbers = list(map(str, numbers))
    line = " & ".join(numbers)
    print(line)
    # pyperclip.copy(line)
