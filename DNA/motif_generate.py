import json
import numpy as np
import argparse
from scipy.stats import bernoulli
from random import randint

Precision = 100000

def generate_from_B(Bn):

    randomNumber = randint(1, Precision)

    if randomNumber <= Bn[0]:
        return 1
    elif (randomNumber > Bn[0]) and (randomNumber <= Bn[1]):
        return 2
    elif (randomNumber > Bn[1]) and (randomNumber <= Bn[2]):
        return 3
    else:
        return 4

def generate_from_T(Tn):

    randomNumber = randint(1, Precision)

    for i in range(0, w-1):
        if randomNumber <= Tn[0][i]:
            return 1
        elif (randomNumber > Tn[0][i]) and (randomNumber <= Tn[1][i]):
            return 2
        elif (randomNumber > Tn[1][i]) and (randomNumber <= Tn[2][i]):
            return 3
        else:
            return 4

def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")

    parser.add_argument('--params', default="", required=True, help='Plik z Parametrami  (default: %(default)s)')
    parser.add_argument('--output', default="", required=True, help='Plik z Parametrami  (default: %(default)s)')
    args = parser.parse_args()

    return args.params, args.output

param_file, output_file = ParseArguments()

with open(param_file, 'r') as inputfile:
    params = json.load(inputfile)

w = params['w']
k = params['k']
alpha = params['alpha']
Theta = np.asarray(params['Theta'])
ThetaB = np.asarray(params['ThetaB'])

Bn = ThetaB * Precision
Bn[1] += Bn[0]
Bn[2] += Bn[1]
Bn[3] += Bn[2]

Tn = Theta * Precision
for i in range(0, w):
    Tn[1][i] += Tn[0][i]
    Tn[2][i] += Tn[1][i]
    Tn[3][i] += Tn[2][i]

Zi = bernoulli.rvs(size=k, p=alpha)

X = np.zeros(shape=(k, w))

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if Zi[i] == 0:
            X[i][j] = generate_from_B(Bn)
        else:
            X[i][j] = generate_from_T(Tn)

gen_data = {
    "alpha": alpha,
    "X": X.tolist()
}

with open(output_file, 'w') as outfile:
    json.dump(gen_data, outfile)