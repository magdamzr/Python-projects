import numpy as np
import pandas as pd
import tarfile
import argparse
from hmmlearn.hmm import GaussianHMM
from math import log

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")

    parser.add_argument('--train', default='', required=True, help='Train file')
    parser.add_argument('--test', default='', required=True, help='Test file')
    parser.add_argument('--output', default='', required=True, help='Result file')

    args = parser.parse_args()
    return args.train, args.test, args.output


train_file, test_directory, output = ParseArguments()

train_data = pd.read_csv(train_file)
appliances = train_data.columns.values.tolist()
appliances.remove('time')
df = np.array(train_data)

test_files = {}
tar = tarfile.open(test_directory, "r:gz")
i = 0
name_list = []
member_list = tar.getmembers()
member_list = sorted(member_list, key=lambda m: m.name)
for member in member_list:
    f = tar.extractfile(member)
    if f is not None:
        i += 1
        content = pd.read_csv(f)
        test_files[i] = content
        name_list.append(member.name[12:])

logLik1 = []
logLik2 = []
logLik3 = []
logLik4 = []
logLik5 = []
AIC1 = []
AIC2 = []
AIC3 = []
AIC4 = []
AIC5 = []
BIC1 = []
BIC2 = []
BIC3 = []
BIC4 = []
BIC5 = []
RIC1 = []
RIC2 = []
RIC3 = []
RIC4 = []
RIC5 = []

for i in range(2, 9):
    p = i**2 + 2*i - 1
    hmm = GaussianHMM(n_components=i)

    X1 = np.column_stack([df[:, 1]])
    model1 = hmm.fit(X1)
    BIC1.append(-2 * model1.score(X1) + p * log(len(X1)))

    # logLik1.append(model1.score(X1))
    # AIC1.append(-2 * model1.score(X1) + 2 * p)
    # RIC1.append(-2 * model1.score(X1) + 2 * p * log(p - 1))

    X2 = np.column_stack([df[:, 2]])
    model2 = hmm.fit(X2)
    BIC2.append(-2 * model2.score(X2) + p * log(len(X2)))

    # logLik2.append(model2.score(X2))
    # AIC2.append(-2 * model2.score(X2) + 2 * p)
    # RIC2.append(-2 * model2.score(X2) + 2 * p * log(p - 1))

    X3 = np.column_stack([df[:, 3]])
    model3 = hmm.fit(X3)
    BIC3.append(-2 * model3.score(X3) + p * log(len(X3)))

    # logLik3.append(model3.score(X3))
    # AIC3.append(-2 * model3.score(X3) + 2 * p)
    # RIC3.append(-2 * model3.score(X3) + 2 * p * log(p - 1))

    X4 = np.column_stack([df[:, 4]])
    model4 = hmm.fit(X4)
    BIC4.append(-2 * model4.score(X4) + p * log(len(X4)))

    # logLik4.append(model4.score(X4))
    # AIC4.append(-2 * model4.score(X4) + 2 * p)
    # RIC4.append(-2 * model4.score(X4) + 2 * p * log(p - 1))

    X5 = np.column_stack([df[:, 5]])
    model5 = hmm.fit(X5)
    BIC5.append(-2 * model5.score(X5) + p * log(len(X5)))

    # logLik5.append(model5.score(X5))
    # AIC5.append(-2 * model5.score(X5) + 2 * p)
    # RIC5.append(-2 * model5.score(X5) + 2 * p * log(p - 1))

# BIC implementation
index_min1 = np.argmin(BIC1)
index_min2 = np.argmin(BIC2)
index_min3 = np.argmin(BIC3)
index_min4 = np.argmin(BIC4)
index_min5 = np.argmin(BIC5)

# logLike implementation
# index_min1 = np.argmax(logLik1)
# index_min2 = np.argmax(logLik2)
# index_min3 =np.argmax(logLik3)
# index_min4 = np.argmax(logLik4)
# index_min5 = np.argmax(logLik5)

# AIC implementation
# index_min1 = np.argmin(AIC1)
# index_min2 = np.argmin(AIC2)
# index_min3 = np.argmin(AIC3)
# index_min4 = np.argmin(AIC4)
# index_min5 = np.argmin(AIC5)

# RIC implementation
# index_min1 = np.argmin(RIC1)
# index_min2 = np.argmin(RIC2)
# index_min3 = np.argmin(RIC3)
# index_min4 = np.argmin(RIC4)
# index_min5 = np.argmin(RIC5)

new_model1 = GaussianHMM(n_components=index_min1+2)
p1 = (index_min1 + 2)**2 + (index_min1 + 2)*2 - 1

new_model2 = GaussianHMM(n_components=index_min2+2)
p2 = (index_min2 + 2)**2 + (index_min2 + 2)*2 - 1

new_model3 = GaussianHMM(n_components=index_min3+2)
p3 = (index_min3 + 2)**2 + (index_min3 + 2)*2 - 1

new_model4 = GaussianHMM(n_components=index_min4+2)
p4 = (index_min4 + 2)**2 + (index_min4 + 2)*2 - 1

new_model5 = GaussianHMM(n_components=index_min5+2)
p5 = (index_min5 + 2)**2 + (index_min5 + 2)*2 - 1

file = open(output, 'w')

for i in range(1, len(test_files)+1):
    list1 = []

    test_file = test_files[i]
    df = np.array(test_file)
    df = np.delete(df, 0, 1)

    model1_min = new_model1.fit(X1)
    list1.append((-2 * model1_min.score(df) + p1 * log(len(df))))

    # list1.append(model1_min.score(df))
    # list1.append(-2 * model1_min.score(df) + 2 * p1)
    # list1.append(-2 * model1_min.score(df) + 2 * p1 * log(p1 - 1))

    model2_min = new_model2.fit(X2)
    list1.append((-2 * model2_min.score(df) + p2 * log(len(df))))

    # list1.append(model2_min.score(df))
    # list1.append(-2 * model2_min.score(df) + 2 * p2)
    # list1.append(-2 * model2_min.score(df) + 2 * p2 * log(p2 - 1))

    model3_min = new_model3.fit(X3)
    list1.append((-2 * model3_min.score(df) + p3 * log(len(df))))

    # list1.append(model3_min.score(df))
    # list1.append(-2 * model3_min.score(df) + 2 * p3)
    # list1.append(-2 * model3_min.score(df) + 2 * p3 * log(p3 - 1))

    model4_min = new_model4.fit(X4)
    list1.append((-2 * model4_min.score(df) + p4 * log(len(df))))

    # list1.append(model4_min.score(df))
    # list1.append(-2 * model4_min.score(df) + 2 * p4)
    # list1.append(-2 * model4_min.score(df) + 2 * p4 * log(p4 - 1))

    model5_min = new_model5.fit(X5)
    list1.append((-2 * model5_min.score(df) + p5 * log(len(df))))

    # list1.append(model5_min.score(df))
    # list1.append(-2 * model5_min.score(df) + 2 * p5)
    # list1.append(-2 * model5_min.score(df) + 2 * p5 * log(p5 - 1))

    device = np.argmin(list1)
    file.write(name_list[i-1])
    file.write(" ")
    file.write(appliances[device])
    file.write("\n")

print("Everything went fine result is in", output, "file")
