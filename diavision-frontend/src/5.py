t = int(input())

for i in range(t):
    n = int(input())
    l = list(map(int, input().split()))

    for j in range(n):
        if l.count(l[j]) == 1:
            print(l[j])