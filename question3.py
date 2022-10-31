import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# matplotlib inline

# graphing parameters
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (16.0, 10.0)

p = Path(r'/Users/adaates/Desktop/code/hw3-neuralnet-sondande')  # path to files
files = ['q3validation-mnist_5v8.csv-0.1r-12345.csv', 'q3validation-mnist_5v8.csv-0.01r-12345.csv', 'q3validation-mnist_5v8.csv-0.001r-12345.csv']
files = list(files)  # get files
lr = ['0.1', '0.01', '0.001']

# everything for here down, belongs in one Jupyter cell
plt.figure()
index = 0
for f in files:  # iterate through files
    version = "LR:" + lr[index]  # get filename
    df = pd.read_csv(f, dtype={'Epochs': int, 'Accuracy': float})  # create dataframe

    print(df.head())  # this is here to verify df has data; it can be commented out or removed

    plt.plot('Epochs', 'Accuracy', data=df, label=version)  # plot the data from each file

    index += 1

plt.legend(bbox_to_anchor=(1.0, 0.5), loc='center left')
plt.savefig('validation.jpg')  # verify there's plot in the file
plt.show()  # outside the loop
