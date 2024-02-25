import numpy as np
import scipy
import matplotlib.pyplot as plt


def dlqr(F, G, Q, R):
    """Solve the discrete time lqr controller.

    x[k+1] = A x[k] + B u[k]

    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # ref Bertsekas, p.151

    # first, try to solve the discrete time ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(F, G, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(G.T * X * G + R) * (G.T * X * F))

    # eigVals, eigVecs = scipy.linalg.eig(A-B*K)

    return -K


def event(k, a=16, b=0.01):
    return a * (1 - b) ** k


def super_event(x_norm, matrix_norm, k, sigma=0.01):
    l = sigma * x_norm / matrix_norm + event(k)

    return l


def main():
    h = 0.01

    A = np.array([[1, h],
                  [0, 1]])
    Q = np.eye(2)

    B = np.array([[0], [h]])
    R = np.eye(1)

    K = dlqr(A, B, Q, R)

    matrix_norm = np.linalg.norm(B @ K, 2)
    print('matrix_norm:', matrix_norm)

    x = np.array([[1], [-12]])
    x_fic = x
    u = np.array([[0]])

    X1 = []
    X2 = []
    U = []
    EV = []
    e = 0
    refresh = 0
    x_norm = np.linalg.norm(x, 2)

    T = list(range(2000))

    for i in T:

        E = super_event(x_norm, matrix_norm, i)

        if e > E:
            u = K @ x
            x_fic = x

            refresh = refresh + 1

        e = np.linalg.norm(x_fic - x, 2)
        x_norm = np.linalg.norm(x, 2)
        x = A @ x + B @ u

        X1.append(x[0, 0])
        X2.append(x[1, 0])
        EV.append(e)
        U.append(u[0, 0])

    print("Update rate %.2f" % (refresh * 100 / len(T)))

    plt.figure(1)
    plt.plot(X1, 'k')
    plt.plot(X2, 'r')
    plt.title("States")
    plt.grid(True)

    plt.figure(2)
    plt.plot(EV, 'b')
    plt.title("Event Error")
    plt.grid(True)

    plt.figure(3)
    plt.plot(U, 'b')
    plt.title("Control Signal")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()