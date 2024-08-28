import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

BOUNDS = {
    "Number of pregnancies": ([0, 2, 6, float("inf")], ["low", "medium", "high"]),
    "Plasma glucose level": ([0.1, 95, 141, float("inf")], ["low", "medium", "high"]),
    "Diastolic Blood Pressure": (
        [0.1, 80, 90, float("inf")],
        ["normal", "pre-hypertension", "high"],
    ),
    "Body Mass Index": (
        [0.1, 18.5, 25.1, 30.1, float("inf")],
        ["low", "healthy", "obese", "severely-obese"],
    ),
    "Diabetes Pedigree Function": (
        [0.001, 0.42, 0.82, float("inf")],
        ["low", "medium", "high"],
    ),
    "Age (in years)": ([0.1, 41, 60, float("inf")], ["r1", "r2", "r3"]),
}


def clean(raw_filename, training_filename, testing_filename, seed):
    """
    Do the work of cleaning the Pima Indians Diabetes dataset.

    Inputs:
      raw_filename (string): name of the file with the original data
      training_filename (string): name of the file for the
        (cleaned) training data
      testing_filename (string): name of the file for the
        (cleaned) testing data
      seed: seed for splitting the original data into testing and training sets.
    """
    #Drop zero values from certain cols
    clean_data = pd.read_csv(raw_filename)
    cols_to_check_zero = ["Plasma glucose level", "Diastolic Blood Pressure", "Body Mass Index"]
    clean_data = clean_data[~(clean_data[cols_to_check_zero] == 0).any(axis =1)]

    #Convert zeroes into NaN values
    cols_clean_zero = clean_data.columns[1:-1]
    clean_data.loc[:, cols_clean_zero] = clean_data.loc[:, cols_clean_zero].replace(0,np.nan)

    #Transform data into categorical values using pre-defined dictionary called Bounds
    for key, value in BOUNDS.items():
        clean_data[key] = pd.cut(clean_data[key], bins = value[0], right = False, labels = value[1])

    #Eliminate cols skin thickness and insulin
    cols_to_drop = ["Triceps skin fold thickness (mm)", "2-Hour serum insulin (mu U/ml)"]
    clean_data = clean_data.drop(cols_to_drop, axis = 1) 

    #Split into test and train groups
    train_data, test_data = train_test_split(clean_data, train_size = 0.9, random_state = seed)
    train_data.to_csv(training_filename, header = True, index = False)
    test_data.to_csv(testing_filename, header = True, index = False)

def main(args):
    """
    Process the arguments and call clean.
    """

    usage = (
        "usage: python3 {} <raw data filename> <training filename>"
        " <testing filename> [seed]"
    )
    if len(args) < 4 or len(args) > 5:
        print(usage.format(args[0]))
        sys.exit(1)

    try:
        if len(args) == 4:
            seed = None
        else:
            seed = int(args[4])
    except ValueError:
        print(usage)
        print("The seed must be an integer")
        sys.exit(1)

    clean(args[1], args[2], args[3], seed)


if __name__ == "__main__":
    main(sys.argv)