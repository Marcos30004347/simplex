import numpy as np
import sys

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
    for j in N:
        A_[l][j] = A[l][j]/A[l][N[e]]
    
    for j in B:
        A_[l][j] = A[l][j]/A[l][N[e]]

    for idx, j in enumerate(vero[l]):
        vero_[l][idx] = vero[l][idx]/A[l][N[e]]

    A_[l][B[l]] = A_[l][B[l]]/A_[l][N[e]]
    A_[l][N[e]] = 1
    # np.set_printoptions(linewidth=1200)

    # pivot
    for i in range(A.shape[0]):
        if i == l:
            continue

        alpha = A_[i][N[e]]


        b_[i] = b[i] - alpha*b_[l]

        # all basic elements in line i
        for j in V:
            vero_[i][j] = vero[i][j] - alpha*vero_[l][j]

        for j in B:
            A_[i][j] = A[i][j] - alpha*A_[l][j]
        
        A_[i][N[e]] = 0

        # all non-basic elements in line i
        for j in N:
            A_[i][j] = A[i][j] - alpha*A_[l][j]

        A_[i][N[e]] = 0
    
    v_ = v - c[N[e]]*b_[l]

    for j in V:
        certificate_[j] = certificate[j] - c[N[e]]*vero_[l][j]

    for j in N:
        c_[j] = c[j] - c[N[e]]*A_[l][j]
    
    for j in B:
        c_[j] = c[j] - c[N[e]]*A_[l][j]

    c_[N[e]] = 0

    N_[e] = B_[l]
    B_[l] = N[e]
    
    return (vero_, certificate_, V_, N_, B_, A_, b_, c_, v_)


def toFPIForm(A, b, c):
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

# Transform original problem into auxiliar problem
def toAuxForm(vero, certificate, V, N, B, A, c, b, v):
    v_      = np.copy(v)
    V_      = np.copy(V)
    N_      = np.empty([0], dtype=int)#np.copy(N)
    B_      = np.empty([0], dtype=int)#np.copy(B)
    A_      = np.copy(A)
    b_      = np.copy(b)
    vero_   = np.copy(vero)

    certificate_ = np.copy(certificate)

    # invert lines where b[i] < 0
    rows = []
    for i in range(A.shape[0]):
        if b[i] < 0:
            b_[i] = -1*b[i]
            rows.append(i)
            for j in range(A.shape[1]):
                A_[i][j] = -1*A[i][j]

            for j in range(len(vero[i])):
                vero_[i][j] = -1*vero[i][j]
        else:
            for j in range(A.shape[1]):
                A_[i][j] = A[i][j]

    # Add extra columns, so that it will be possible to retrieve a 
    # feasible solution
    A__ = np.zeros((A.shape[0], A.shape[1] + len(rows)))

    c_ = np.zeros(len(c) + len(rows))
    
    for i in range(A_.shape[0]):
        for j in range(A_.shape[1]):
            A__[i][j] = A_[i][j]
    
    # Add 1 to extra columns in the correct positions to result in a
    # feasible solution
    for i in range(len(rows)):
        A__[rows[i]][A.shape[1]+i] = 1

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
        B_ = np.append(B_, [i+len(c)])

    k = 0

    # Set the set of non basic variables
    for i in range(len(c) - len(b)):
        N_ = np.append(N_, [i])

    for i in range(len(B)):
        if A__[i][B[i]] < 0:
            B_[i] = A.shape[1] + k
            k = k+1
            N_ = np.append(N_, [B[i]])
        else:
            B_[i] = B[i]

    # Set elements in c that are in the base but are not 0
    for k in B_:
        if c_[k] != 0:
            for i in range(A__.shape[0]):
                if A__[i][k] != 0:
                    alpha = c_[k]/A__[i][k]
                    v_ -= alpha*b_[i]
                    for j in range(len(c)):
                        c_[j] = c_[j] - A__[i][j]*alpha
                    for j in V:
                        certificate_[j] = certificate_[j] - vero[i][j]*alpha

    return (vero_, certificate_, V_, N_, B_, A__, b_, c_, v_, rows)


# Print System in a readable way
def printSystem(vero, certificate, V, N, B, A, b, c, v):
    print("N =", end=" ")
    for i in N:
        print("x" + str(int(i)), end=" ")
    print(", ", end="")     
    print("B =", end=" ")
    for i in B:
        print("x" + str(int(i)), end=" ")
    print("")
    for j in range(len(certificate)):
        print('{:5}'.format(round(certificate[j],2)), end=" ")
    print('{:5}'.format("||"), end="")
    for j in range(len(c)):
        print('{:5}'.format(round(c[j],2)), end=" ")
    print('{:5}'.format("="), end=" ")
    print('{:5}'.format(round(v,2)), end="")
    print()
    for j in range((len(certificate) + len(c))*2):
        print("----", end="")
    print()
    for i in range(A.shape[0]):
        for j in range(vero.shape[1]):
            print('{:5}'.format(round(vero[i][j],2)), end=" ")
        print('{:5}'.format("||"), end="")
        for j in range(A.shape[1]):
            print('{:5}'.format(round(A[i][j],2)), end=" ")
        print('{:5}'.format("="), end=" ")
        print('{:5}'.format(round(b[i],2)))
    print()
    print()


# Choose column that will enter the base, the first negative element in c
def pickColumn(c, N):
    for i, _ in enumerate(N):
        if c[N[i]] < 0:
            return i
    return 0

# Solve a linear problem c.T*x with constraints A*x <= b
def solve(A, b, c):
    initial_variables = len(c)
    # To equalitys form
    vero, certificate, V, N, B, A, b, c, v = toFPIForm(A, b, c)
    print("FPI: ")
    printSystem(vero, certificate, V, N, B, A, b, c, v)

    C = np.copy(c)

    # Solve auxiliary problem
    vero_aux, certificate_aux, V_aux, N_aux, B_aux, A_aux, b_aux, c_aux, v_aux, slack = toAuxForm(vero, certificate, V, N, B, A, c, b, v)
    print("Aux: ")
    printSystem(vero_aux, certificate_aux, V_aux, N_aux, B_aux, A_aux, b_aux, c_aux, v_aux)
    vero, certificate, V, N, B, A, b, c, v = simplex(vero_aux, certificate_aux, V_aux, N_aux, B_aux, A_aux, b_aux, c_aux, v_aux)

    # problem is infeasible
    if round(v, 4) != 0:
        print("inviavel")
        for xi in certificate:
            print( ('%f' % round(xi, 7)).rstrip('0').rstrip('.'), end=" ")
        print()
        quit()

    # Remove auxiliary columns
    for k in range(len(slack)):
        A = np.delete(A, len(c) - k - 1, 1)
        N = np.delete(N, np.where(N == len(c) - k - 1))
        B = np.delete(B, np.where(B == len(c) - k - 1))
    
    # Remove linearly dependent lines
    # i = 0
    # while i < len(b):
    #     if b[i] == 0:
    #         A = np.delete(A, i, 0)
    #         b = np.delete(b, i, 0)
    #         vero = np.delete(vero, i, 0)
    #         continue
    #     i = i+1

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
                    v -= alpha*b[i]
                    for j in range(len(C)):
                        C[j] = C[j] - A[i][j]*alpha
                    for j in V:
                        certificate[j] = certificate[j] - vero[i][j]*alpha


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
        print( ('%f' % round(xi, 7)).rstrip('0').rstrip('.'), end=" ")
    print()
    for xi in certificate:
        print( ('%f' % round(xi, 7)).rstrip('0').rstrip('.'), end=" ")
    print()

def simplex(vero, certificate, V, N, B, A, b, c, v):
    while (np.take(c, N) < 0).sum():
        # Round values to precision
        c = np.round(c, 10)
        b = np.round(b, 10)
        A = np.round(A, 10)

        vero = np.round(vero, 10)
        certificate = np.round(certificate, 10)
        
        # choose column that will enter the base
        e = pickColumn(c, N)

        delta = np.ones(len(b)) * np.inf

        # this should be for each variable in base
        for i in range(A.shape[0]):
            if A[i][N[e]] > 0:
                delta[i] = b[i]/A[i][N[e]]

        l = np.argmin(delta)

        if delta[l] == np.Inf:
            print("ilimitada")

            certificate = np.zeros(len(c) - len(b))

            certificate[l] = 1

            # Build certificate
            k = 0
            for i in B:
                if k == l:
                    k = k + 1
                for j in range(A.shape[0]):
                    if A[j][i] != 0:    
                        certificate[k] = A[j][l]*-1
                        k = k + 1
            
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
                print( ('%f' % round(xi, 7)).rstrip('0').rstrip('.'), end=" ")
            print()

            for xi in certificate:
                print( ('%f' % round(xi, 7)).rstrip('0').rstrip('.'), end=" ")
            print()
            quit()
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