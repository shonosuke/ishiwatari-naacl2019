# stdin : train.log
# stdout:  test.out
import sys

in_test_data = False
for line in sys.stdin:
    if in_test_data:
        if line[0] == '=':
            break
        else:
            print(line, end='')
    elif line == 'Decoded:\n':
        in_test_data = True
