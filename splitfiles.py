

with open('rt-polarity.neg') as sourcefile:
    for i, line in enumerate(sourcefile):
        with open("rt-polarity {}.neg".format(str(i+1)), "w") as txtfile:
            txtfile.write(line)