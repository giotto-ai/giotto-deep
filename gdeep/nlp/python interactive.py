# %%
1+1
# %%


class Test():
    def __init__(self):
        print("class constructed")

t = Test()


def sum(x):
    sum = 0
    for i in x:
        sum += i
    return sum


sum([1, 2, 3])

t
# %%

a = 1
b = 2

diff = a - b


