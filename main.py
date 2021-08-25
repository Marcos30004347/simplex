import numpy as np
import sys

from numpy.core.numeric import isclose

def pivot(vero, cert, V, N, B, A, b, c, v, l, e):
    b[l]       = b[l] / A[l][N[e]]
    vero[l][:] = vero[l][:] / A[l][N[e]]
    A[l][:]    = A[l][:] / A[l][N[e]]

    for i in range(A.shape[0]):
        if i == l: continue

        alpha      = A[i][N[e]]
        b[i]       = b[i] - alpha*b[l]
        vero[i][:] = vero[i][:] - alpha*vero[l][:]
        A[i][:]    = A[i][:] - alpha*A[l][:]

    # print(c[N[e]]*b[l])

    v       = v - c[N[e]]*b[l]
    cert[:] = cert[:] - c[N[e]]*vero[l][:]
    c[:]    = c[:] - c[N[e]]*A[l][:]
    tmp     = B[l]
    B[l]    = N[e]
    N[e]    = tmp
    
    return (vero, cert, V, N, B, A, b, c, v)


def toFPIForm(A, b, c):
    slack_count = len(b)

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

    b_ = np.copy(b)

    # A_[0][columns] = 0
    # print(columns)
    # B_ = np.delete(B_, np.where(B_ == columns))
    # print(B_)

    v_ = 0

    return (slack_count, vero_, certificate_, V_, np.sort(N_), np.sort(B_), A_, b_, c_, v_)

# Transform original problem into auxiliar problem
def toAuxForm(vero, certificate, V, N, B, A, c, b, v):
    v_      = np.copy(v)
    V_      = np.copy(V)
    N_      = np.empty([0], dtype=int)#np.copy(N)
    B_      = np.empty([0], dtype=int)#np.copy(B)
    b_      = np.copy(b)
    vero_   = np.copy(vero)

    certificate_ = np.copy(certificate)

    # Add extra columns, so that it will be possible to retrieve a 
    # feasible solution
    A_ = np.zeros((A.shape[0], A.shape[1] + len(b)))

    # Add 1 to extra columns in the correct positions to result in a
    # feasible solution
    for i in range(len(b)):
        A_[i][A.shape[1] + i] = 1

    # invert lines where b[i] < 0
    for i in range(A.shape[0]):
        if b[i] < 0:
            b_[i] = -1*b[i]

            for j in range(A.shape[1]):
                A_[i][j] = -1*A[i][j]

            for j in range(len(vero[i])):
                vero_[i][j] = -1*vero[i][j]
        else:
            for j in range(A.shape[1]):
                A_[i][j] = A[i][j]

    c_ = np.zeros(len(c) + len(b))
    
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

    # Set the set of basic variables
    for i in range(len(b)):
        B_ = np.append(B_, [i + len(c)])

    # Set the set of non basic variables
    for i in range(len(c_) - len(b)):
        N_ = np.append(N_, [i])

    return (vero_, certificate_, V_, N_, B_, A_, b_, c_, v_)


# Print System in a readable way
def printSystem(vero, certificate, V, N, B, A, b, c, v):
    print("Non-Basic =", end=" ")
    for i in N:
        print("x" + str(int(i)), end=" ")
    print()
    print("Basic     =", end=" ")
    for i in B:
        print("x" + str(int(i)), end=" ")
    print("")
    print()
    for j in range(len(certificate)):
        print('{:7}'.format(round(certificate[j],2)), end=" ")
    print('{:7}'.format("||"), end="")
    for j in range(len(c)):
        print('{:7}'.format(round(c[j],2)), end=" ")
    print('{:7}'.format("="), end=" ")
    print('{:7}'.format(round(v,2)), end="")
    print()
    # for j in range((len(certificate) + len(c) + 3)):
    #     print('{:7}'.format("*"), end=" ")
    print()
    for i in range(A.shape[0]):
        for j in range(vero.shape[1]):
            print('{:7}'.format(round(vero[i][j],2)), end=" ")
        print('{:7}'.format("||"), end="")
        for j in range(A.shape[1]):
            print('{:7}'.format(round(A[i][j],2)), end=" ")
        print('{:7}'.format("="), end=" ")
        print('{:7}'.format(round(b[i],2)))
    print()
    print()


# Choose column that will enter the base, the first negative element in c
def pickColumn(c, N):
    idx = np.inf

    for i, _ in enumerate(N):
        if(np.isclose(c[N[i]], 0)):
            c[N[i]] = 0

        if c[N[i]] < 0 and (idx == np.inf or N[i] < N[idx]):
            idx = i

    if(idx == np.inf): return -1
    return idx

def unbouded(N, B, A, c, b, e):
    print("ilimitada")

    certificate = np.zeros(len(c) + len(b))
    
    col = N[e]
    certificate[col] = 1

    # Build certificate
    k = 0
    for i in range(len(c)):
        if(k == col):
            k = k + 1
            continue
        if i in B:
            for q in range(A.shape[0]):
                if A[q][i] != 0:  
                    certificate[k] = A[q][N[e]]*-1
        else:
            certificate[k] = 0
        k = k + 1

    certificate = certificate[:len(c) - len(b)]

    # Build feasible solution
    x = np.zeros(len(c) - len(b))
    for i in range(len(x)):
        if(c[i] != 0):
            x[i] = 0
        else:
            for j in range(A.shape[0]):
                if(A[j][i] == 1):
                    x[i] = b[j]
    for xi in x:
        if(np.isclose(xi, 0)): xi = 0
        print( ('%f' % round(xi, 7)).rstrip('0').rstrip('.'), end=" ")
    print()

    for xi in certificate:
        if(np.isclose(xi, 0)): xi = 0
        print( ('%f' % round(xi, 7)).rstrip('0').rstrip('.'), end=" ")
    print()
    quit()

# Solve a linear problem c.T*x with constraints A*x <= b
def solve(A, b, c):
    initial_variables = len(c)
    # To equalitys form
    slack_count, vero, certificate, V, N, B, A, b, c, v = toFPIForm(A, b, c)

    printSystem(vero, certificate, V, N, B, A, b, c, v)

    C = np.copy(c)

    # Solve auxiliary problem
    vero_aux, certificate_aux, V_aux, N_aux, B_aux, A_aux, b_aux, c_aux, v_aux = toAuxForm(vero, certificate, V, N, B, A, c, b, v)
    
    printSystem(vero_aux, certificate_aux, V_aux, N_aux, B_aux, A_aux, b_aux, c_aux, v_aux)
    
    vero, certificate, V, N, B, A, b, c, v = simplex(vero_aux, certificate_aux, V_aux, N_aux, B_aux, A_aux, b_aux, c_aux, v_aux)

    # problem is infeasible
    if round(v, 4) != 0:
        print("inviavel")
        for xi in certificate:
            if(np.isclose(xi, 0)): xi = 0
            print( ('%f' % round(xi, 7)).rstrip('0').rstrip('.'), end=" ")
        print()
        quit()

    # Remove auxiliary columns
    for k in range(len(b)):
        A = np.delete(A, len(c) - k - 1, 1)
        N = np.delete(N, np.where(N == len(c) - k - 1))
        B = np.delete(B, np.where(B == len(c) - k - 1))
    
    # Reset certificate
    for j in range(len(certificate)):
        certificate[j] = 0

    # Pivoting columns k that are on feaseble solution but c[k] is not 0
    v = 0
    for k in B:
        if C[k] != 0:
            for i in range(A.shape[0]):
                if A[i][k] != 0.0:
                    alpha = C[k]/A[i][k]
                    v = v - alpha*b[i]
                    C[:] = C[:] - A[i][:]*alpha
                    certificate[:] = certificate[:] - vero[i][:]*alpha

    printSystem(vero, certificate, V, N, B, A, b, C, v)
    
    # Solve resulting system
    vero, certificate, V, N, B, A, b, c, v = simplex(vero, certificate, V, N, B, A, b, C, v)
    
    # printSystem(vero, certificate, V, N, B, A, b, C, v)
    x_ = np.zeros(len(c))
    np.set_printoptions(linewidth=1200)


    for i, _ in enumerate(B):
        x_[B[i]] = b[i]

    x = np.empty([0])
    for i in range(initial_variables):
        x = np.append(x, x_[i])
    
    print("otima")
    print( ('%f' % round(v, 7)).rstrip('0').rstrip('.'))
    for xi in x:
        if(np.isclose(xi, 0)): xi = 0
        print( ('%f' % round(xi, 7)).rstrip('0').rstrip('.'), end=" ")
    print()
    for xi in certificate:
        if(np.isclose(xi, 0)): xi = 0
        print( ('%f' % round(xi, 7)).rstrip('0').rstrip('.'), end=" ")
    print()

def simplex(vero, certificate, V, N, B, A, b, c, v):
    while (np.take(c, N) < 0).sum():

        # choose column that will enter the base
        e = pickColumn(c, N)
        
        if e == -1:
            break

        delta = np.ones(len(b)) * np.inf

        for i in range(A.shape[0]):
            if A[i,N[e]] > 0:
                delta[i] = b[i]/A[i,N[e]]
        
        l = 0

        # Choose line using Bland's rule
        for i in range(A.shape[0]):
            if(delta[i] == np.inf):
                continue

            if(delta[l] > delta[i]):
                l = i

            # Bland's rule
            if np.isclose(delta[l], delta[i]):
                if b[i] < b[l]:
                    l = i
        
        A = np.reshape(A, A.shape)
    
        if delta[l] == np.Inf:
            unbouded(N, B, A, c, b, e)
        else:
            vero, certificate, V, N, B, A, b, c, v = pivot(vero, certificate, V, N, B, A, b, c, v, l, e)
            printSystem(vero, certificate, V, N, B, A, b, c, v)



    return vero, certificate, V, N, B, A, b, c, v

c = []
A = []

row = 0

first_line = input().split()

n = int(first_line[0])
m = int(first_line[1])

second_line = input().split()

c = np.zeros(m)

for i in range(len(second_line)):
    c[i] = int(second_line[i])

A = np.zeros((n, m))

b = np.zeros(n)

row = 0

for line_str in sys.stdin:
    line = line_str.split()
    for i in range(m):
        A[row][i] = int(line[i])
    b[row] = int(line[m])
    row = row + 1

solve(A, b, c)