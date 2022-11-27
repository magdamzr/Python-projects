import json
import numpy as np
import argparse
from scipy.stats import bernoulli

def probability_Theta(k):

    prob = 1
    for i in range(w):
        prob *= Theta[int(X[k][i]) - 1, i]
    return prob

def probability_ThetaB(k):

    prob = 1
    for i in range(w):
        prob *= ThetaB[int(X[k][i]) - 1]
    return prob

def EM (pX):

    Qi_ThetaB = [(1-alpha)*probability_ThetaB(i) / ((1 - alpha)*probability_ThetaB(i) + alpha*probability_Theta(i)) for i in range(k)]
    Qi_Theta = [alpha*probability_Theta(i) / (alpha*probability_Theta(i) + (1-alpha)*probability_ThetaB(i)) for i in range(k)]

    new_XB = np.zeros(pX.shape)
    for i in range(k):
        new_XB[i] = pX[i] * Qi_ThetaB[i]

    new_X = np.zeros(pX.shape)
    for i in range(k):
        new_X[i] = pX[i] * Qi_Theta[i]

    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    for i in range(k):
        s1 += Qi_ThetaB[i] * np.sum(pX[i] == 1)
        s2 += Qi_ThetaB[i] * np.sum(pX[i] == 2)
        s3 += Qi_ThetaB[i] * np.sum(pX[i] == 3)
        s4 += Qi_ThetaB[i] * np.sum(pX[i] == 4)
        pass

    new_ThetaB = [s1 / (w * sum(Qi_ThetaB)), s2 / (w * sum(Qi_ThetaB)), s3 / (w * sum(Qi_ThetaB)), s4 / (w * sum(Qi_ThetaB))]

    new_Theta = np.zeros(Theta.shape)
    new_Theta2 = np.zeros(Theta.shape)

    for i in range(w):
        for j in range(k):
            value = X[j][i]
            value = value.astype(np.int)
            new_Theta[value - 1][i] += value * Qi_Theta[j]

    for i in range(w):
        new_Theta2[:, i] = new_Theta[:, i] / sum(new_Theta[:, i])

    return new_Theta2, new_ThetaB

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")

    parser.add_argument('--input', default="", required=True, help='Plik z danymi  (default: %(default)s)')
    parser.add_argument('--output', default="", required=True, help='Tutaj zapiszemy wyestymowane parametry  (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False, help='Czy estymowac alpha czy nie?  (default: %(default)s)')
    args = parser.parse_args()

    return args.input, args.output, args.estimate_alpha

input_file, output_file, estimate_alpha = ParseArguments()

with open(input_file, 'r') as inputfile:
    data = json.load(inputfile)
alpha = data['alpha']
X = np.asarray(data['X'])
k, w = X.shape

Zi = bernoulli.rvs(size=k, p=alpha)

ThetaT = np.zeros((w, 4))
ThetaB = np.zeros(4)

counterT = 0
counterTB = 0

for i in range(0, k):
    for j in range(0, w):

        pos = X[i][j]
        intPos = pos.astype(np.int)

        if Zi[i] == 1:
            ThetaT[j][intPos - 1] += 1
            counterT += 1
        else:
            ThetaB[intPos - 1] += 1
            counterTB += 1

for i in range(0, w):
    colSum = ThetaT[i][0] + ThetaT[i][1] + ThetaT[i][2] + ThetaT[i][3]
    ThetaT[i][0] /= colSum
    ThetaT[i][1] /= colSum
    ThetaT[i][2] /= colSum
    ThetaT[i][3] /= colSum

for i in range(0, 4):
    ThetaB[i] /= counterTB

Theta = np.transpose(ThetaT)

maxIter = 500
numIter = 0

ThetaOld = Theta
ThetaBOld = ThetaB

result = EM(X)
Theta = result[0]
ThetaB = result[1]
ThetaB = np.asarray(ThetaB)

while True:
    if ((np.sum((ThetaOld - Theta)**2) < 1e-16 and np.sum((ThetaBOld - ThetaB)**2) < 1e-16) or numIter == maxIter):
        break
    else:
        ThetaOld = Theta
        ThetaBOld = ThetaB

        result = EM(X)
        Theta = result[0]
        ThetaB = result[1]
        ThetaB = np.asarray(ThetaB)

    numIter = numIter + 1

ThetaB = np.asarray(ThetaB)

estimated_params = {
    "alpha": alpha,
    "Theta": Theta.tolist(),
    "ThetaB": ThetaB.tolist()
}

with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)