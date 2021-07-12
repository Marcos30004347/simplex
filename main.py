import numpy as np
import sys

def pivot(N, B, A, b, c, v, l, e):
    # print(B[base_idx], ", ", N[e])
    # print("N =", end=" ")
    # for i in N:
    #     print("x" + str(int(i)), end=" ")
    # print("")     
    
    # print("B =", end=" ")
    # for i in B:
    #     print("x" + str(int(i)), end=" ")
    # print("")

    A_ = np.copy(A)
    b_ = np.copy(b)
    c_ = np.copy(c)
    v_ = 0

    # divide pivoting line by the pivoted element
    b_[l] = b[l]/A[l][N[e]]

    for j in [x for idx, x in enumerate(N) if x != N[e]]:
        A_[l][j] = A[l][j]/A[l][N[e]]

    for j in [x for idx, x in enumerate(B) if x != B[l]]:
        A_[l][j] = A[l][j]/A[l][N[e]]

    A_[l][B[l]] = A_[l][B[l]]/A_[l][N[e]]
    A_[l][N[e]] = 1

    # pivot
    for i in range(A.shape[0]):
        if i == l:
            continue

        # all basic elements in line i
        for j in B:
            A_[i][j] = A[i][j] - A_[l][j]
        A_[i][N[e]] = 0

        # all non-basic elements in line i
        for j in [x for idx, x in enumerate(N) if x != e]:
            A_[i][j] = A[i][j] - A_[l][j]
        A_[i][N[e]] = 0

    v_ = v + c[N[e]] * b_[l]

    for j in [x for idx, x in enumerate(N) if x != e]:
        c_[j] = c[j] - c[N[e]]*A_[l][j]

    for j in B:
        c_[j] = c[j] - c[N[e]]*A_[l][j]

    c_[N[e]] = 0
    
    N_ = np.copy(N)#np.append([x for idx, x in enumerate(N) if x != N[e]], B[l])
    B_ = np.copy(B)#np.append([x for idx, x in enumerate(N) if x != N[e]], B[l])
    N_[e] = B_[l]
    B_[l] = N[e]
    # B_ = np.append([x for idx, x in enumerate(B) if x != B[l]], N[e])

    return (N_, B_, A_, b_, c_, v_)

def to_slack_form(A, b, c):
    c_ = np.append(c, np.zeros(len(b)))
    A_ = np.zeros((A.shape[0], A.shape[1] + A.shape[0]))

    rows, columns = A.shape
    
    B_ = np.empty([0], dtype=int)
    N_ = np.empty([0], dtype=int)

    for i in range(columns):
        N_ = np.append(N_, [i])

    for r in range(rows):
        B_ = np.append(B_, [rows+r])
        row = np.copy(A[r,:])
        ide = np.zeros((len(b)))
        ide[r] = 1
        for i in range(rows):
            A_[r][i] = row[i]
        for i in range(len(b)):
            A_[r][rows + i] = ide[i]
        # A_.append(np.append(row, ide))
        
    b_ = np.copy(b)

    v_ = 0

    return (np.sort(N_), np.sort(B_), A_, b_, c_, v_)

class Unbounded:
    pass

def print_slack_form(N, B, A, b, c, v):
    print("v =", v)

    print("N =", end=" ")
    for i in N:
        print("x" + str(int(i)), end=" ")
    print("")     
    
    print("B =", end=" ")
    for i in B:
        print("x" + str(int(i)), end=" ")
    print("")
    
    print("")
    rows, columns = A.shape

    for i in c:
        print(i, end=" ")
    print("")
    for i in range(rows):
        for j in range(columns):
            print(A[i][j], end=" ")
        print("=", b[i])
    print()

def pick_column(A, c, N):
    return np.argmax(c)
    best_Val = -np.Inf
    best = np.Inf
    for col in N:
        if(c[col] > 0):
            for r in range(A.shape[0]):
                if(A[r][col] > 0 and c[col] > best_Val):
                    best = col
                    best_Val = c[col]
                    break
    if best_Val == -np.Inf:
        return -1
    
    return best

def simplex(A, b, c):
    N, B, A, b, c, v = to_slack_form(A, b, c)

    print_slack_form(N, B, A, b, c , v)

    it = 0

    while (np.take(c, N) > 0).sum():
        # e = next(x for x, val in enumerate(c) if val > 0)
        e = pick_column(A, c, N)

        delta = np.ones(len(c)) * np.inf
    
        # this should be for each variable in base
        for i in range(A.shape[0]):
            if A[i][N[e]] > 0:
                delta[i] = b[i]/A[i][N[e]]

        l = np.argmin(delta)

        if delta[l] == np.Inf:
            raise Unbounded
        else:
            print("entering x" + str(int(N[e])))
            print("leaving x" + str(int(B[l])))
            N, B, A, b, c, v = pivot(N, B, A, b, c, v, l, e)
            print_slack_form(N, B, A, b, c, v)
            it = it+1
            if(it >= 3):
                break
    
    x_ = np.zeros(len(c))

    # for i in range(len(c)):
    #     if i in B[:]:
    #         x_[i] = b[i]
    
    return x_

c = []
A = []

row = 0

first_line = input().split(" ")

n = int(first_line[0])
m = int(first_line[1])

# print(n,  m)

second_line = input().split(" ")

c = np.zeros(m)

for i in range(len(second_line)):
    c[i] = int(second_line[i])

# print(c)

A = np.zeros((n, m))

b = np.zeros(n)

row = 0

for line_str in sys.stdin:
    line = line_str.split(" ")
    for i in range(m):
        A[row][i] = int(line[i])
    b[row] = int(line[m])
    row = row + 1

# print(A)
# print(b)

simplex(A, b, c)