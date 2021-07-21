import numpy as np
import sys

class Unbounded:
    pass

# V = vero columns of vero
# N = non-basic columns of A
# B = basic columns of A
def pivot(vero, certificate, V, N, B, A, b, c, v, l, e):
    
    vero_ = np.copy(vero)
    certificate_ = np.copy(certificate)
    
    A_ = np.copy(A)
    V_ = np.copy(V)
    N_ = np.copy(N)
    B_ = np.copy(B)
    b_ = np.copy(b)
    c_ = np.copy(c)
    v_ = 0
    
    # divide pivoting line by the pivoted element
    b_[l] = b[l]/A[l][N[e]]
    for j in [x for idx, x in enumerate(N) if x != N[e]]:
        A_[l][j] = A[l][j]/A[l][N[e]]
    
    for j in [x for idx, x in enumerate(B) if x != B[l]]:
        A_[l][j] = A[l][j]/A[l][N[e]]

    for idx, j in enumerate(vero[l]):
        vero_[l][idx] = vero[l][idx]/A[l][N[e]]

    A_[l][B[l]] = A_[l][B[l]]/A_[l][N[e]]
    A_[l][N[e]] = 1

    # pivot
    for i in range(A.shape[0]):
        if i == l:
            continue

        alpha = A_[i][N[e]]

        # print("alpha", alpha)
        # print("b[i]", b[i])
        # print("alpha*b_[l]", alpha*b_[l])
        b_[i] = b[i] - alpha*b_[l]

        # all basic elements in line i
        for j in V:
            vero_[i][j] = vero[i][j] - alpha*vero_[l][j]

        for j in B:
            A_[i][j] = A[i][j] - alpha*A_[l][j]
        
        A_[i][N[e]] = 0

        # all non-basic elements in line i
        for j in [x for idx, x in enumerate(N) if x != e]:
            A_[i][j] = A[i][j] - alpha*A_[l][j]
        A_[i][N[e]] = 0

    v_ = v - c[N[e]]*b_[l]

    for j in V:
        certificate_[j] = certificate[j] - c[N[e]]*vero_[l][j]
    
    for j in [x for idx, x in enumerate(N) if x != e]:
        c_[j] = c[j] - c[N[e]]*A_[l][j]

    for j in B:
        c_[j] = c[j] - c[N[e]]*A_[l][j]

    c_[N[e]] = 0

    N_[e] = B_[l]
    B_[l] = N[e]

    return (vero_, certificate_, V_, N_, B_, A_, b_, c_, v_)

def to_slack_form(A, b, c):
    c_ = np.append(c*-1, np.zeros(len(b)))
    A_ = np.zeros((A.shape[0], A.shape[1] + A.shape[0]))

    vero_ = np.zeros((A.shape[0], A.shape[0]))
    certificate_ = np.zeros(len(b))

    rows, columns = A.shape

    V_ = np.empty([0], dtype=int)
    B_ = np.empty([0], dtype=int)
    N_ = np.empty([0], dtype=int)

    for i in range(len(b)):
        V_ = np.append(V_, [i])

    for i in range(columns):
        N_ = np.append(N_, [i])

    for r in range(rows):
        B_ = np.append(B_, [columns + r])
        row = np.copy(A[r,:])
        ide = np.zeros((len(b)))
        ide[r] = 1

        for i in range(columns):
            A_[r][i] = row[i]

        for i in range(len(b)):
            A_[r][columns + i] = ide[i]
            vero_[r][i] = ide[i]

        # A_.append(np.append(row, ide))
        
    b_ = np.copy(b)

    v_ = 0

    return (vero_, certificate_, V_, np.sort(N_), np.sort(B_), A_, b_, c_, v_)

def to_auxiliar(vero, certificate, V, N, B, A, c, b, v):
    v_ = np.copy(v)
    V_ = np.copy(V)
    N_ = np.copy(N)
    B_ = np.copy(B)
    A_ = np.copy(A)
    b_ = np.copy(b)

    vero_ = np.copy(vero)
    certificate_ = np.copy(certificate)
    c_ = np.append(np.zeros(len(c)), np.zeros(len(b)))

    need_auxiliar = False

    # invert lines where b[i] < 0
    for i in range(len(b)):
        if b[i] < 0:
            need_auxiliar = True
            b_[i] = -1*b[i];
            for j in range(len(A[i])):
                A_[i][j] = -1*A[i][j]
            for j in range(len(vero[i])):
                vero_[i][j] = -1*vero[i][j]

    if not need_auxiliar:
        return (vero, certificate, V, np.sort(N), np.sort(B), A, b, c, v)
    
    # Subtract lines of A from c
    for j in range(len(c)):
        for i in range(len(b)):
            c_[j] -= A_[i][j]

    # Subtract lines of vero from the certificate
    for j in V:
        for i in range(len(b)):
            certificate_[j] -= vero_[i][j]
    
    # Subtract the b' vector from the optimal value
    for i in range(len(b)):
        v_ = v_ - b_[i]

    return (vero_, certificate_, V_, np.sort(N_), np.sort(B_), A_, b_, c_, v_)


def print_slack_form(vero, certificate, V, N, B, A, b, c, v):
    print("V =", end=" ")
    for i in V:
        print("x" + str(int(i)), end=" ")
    print("")
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

    for i in certificate:
        print('%.2f'%i, end=" ")

    for i in c:
        print('%.2f'%i, end=" ")

    print('= %.2f'%v)


    for i in range(rows):
        for j in vero[i]:
            print('%.2f'%j, end=" ")
        for j in range(columns):
            print('%.2f'%A[i][j], end=" ")
        print("=", '%.2f'%b[i])
    print()


def pick_column(A, c, N):
    print(np.argmin(c))
    return np.argmin(c)

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
    vero, certificate, V, N, B, A, b, c, v = to_slack_form(A, b, c)
    print_slack_form(vero, certificate, V, N, B, A, b, c , v)

    vero, certificate, V, N, B, A, b, c, v = to_auxiliar(vero, certificate, V, N, B, A, c, b, v)

    print_slack_form(vero, certificate, V, N, B, A, b, c , v)

    it = 0
    # print(N - len(b))
    while (np.take(c, N) < 0).sum():
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
            vero, certificate, V, N, B, A, b, c, v = pivot(vero, certificate, V, N, B, A, b, c, v, l, e)
            print_slack_form(vero, certificate, V, N, B, A, b, c, v)

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