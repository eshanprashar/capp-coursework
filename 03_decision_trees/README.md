# Decision Tree Implementation

## About the Data

The Pima Indian Data Set, which is from the
[UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes),
contains anonymized information on women from the
[Pima Indian Tribe](https://en.wikipedia.org/wiki/Pima_people).

This information was collected by the National Institute of Diabetes and Digestive and Kidney Diseases to study diabetes, which is prevalent among members of this tribe.

The data set has 768 entries, each of which contains the following attributes:

* Number of pregnancies
* Plasma glucose concentration from a 2 hours in an oral glucose tolerance test
* Diastolic blood pressure (mm Hg)
* Triceps skin fold thickness (mm)
* 2-Hour serum insulin (mu U/ml)
* Body mass index (weight in kg/(height in m)^2)
* Diabetes pedigree function
* Age (years)
* Has diabetes (1 means yes, 0 means no)

## Task 1: Data Cleaning

Your first task is to clean and then transform the raw Pima data into a training set and a testing set.

We have seeded your directory with a file named
``transform.py``.  Your task is to complete the function ``clean``, which takes four arguments:

* the name of a raw Pima Indians Diabetes data file
* a filename for the training data
* a filename for the testing data
* seed for use with ``train_test_split``

Your function should clean and transform the raw data as described below, split the resulting data into training and testing sets, and save the split data in CSV files.

Each row in the raw file contains an observation.  The raw attribute values are floating point numbers.
For every attribute but the first and the last, a zero should be interpreted as missing data.
The ``"Triceps skin fold thickness (mm)"`` and ``"2-Hour serum insulin (mu U/ml)"`` columns have a lot of missing data, so you should eliminate them when you process the data.
Also, you should remove any observation with a value of zero for plasma glucose concentration, diastolic blood pressure, or body mass index.

Once the data is cleaned, you will need to convert the numeric data into categorical data.
We have included a dictionary named ``BOUNDS``
in ``transform.py``, that specifies for each category, a list of category boundaries and a list of category labels.
For example, the categories for plasma glucose level are represented with the following
pair of lists: ``[0.1, 95, 141, float("inf")]`` and ``["low",
"medium", "high"]``.

Together, these list specify that a plasma
glucose level between 0.1 (inclusive) and 95 (exclusive) should be
labeled as "low", a level between 95 (inclusive) and 141 (exclusive)
should be labeled as medium, and a level of 141 or higher should be
labeled as high.

Note: the Python expression ``float('inf')`` evaluates to positive infinity.  For all floating point values ``x``, ``x < float('inf')``.

Finally, once the data is cleaned and transformed, you should randomly split the observations into two sets--training and testing--using the
``sklearn.model_selection`` function ``train_test_split`` using the specified seed for the ``random_state`` parameter.
The training set should contain roughly 90% of the transformed data, with the remainder going into the testing
set.

The raw data includes a header row, which should be suitably modified and included in both output files.  *Do not include the row index in the output files.*

Pandas is ideally suited for this task.

**Testing Task 1**

 We have provided test code for Task 1 in ``test_transform.py``.

## Decision Trees

As we discussed in lecture, decision trees are a data structure used to solve classification problems.

 Here is a sample decision tree that
labels tax payers as potential cheaters or non-cheaters.

![](sample-tree.png)

This tree, for example, would classify a single person who did not get a refund and makes $85,000 a year as a possible cheater.

We briefly summarize the algorithm for building decision trees below.

See the chapter on [Classification and Decision Trees](https://www-users.cs.umn.edu/~kumar001/dmbook/ch3_classification.pdf) from
*Introduction to Data Mining* by Tan, Steinbach, and Kumar for a more detailed description.

### Definitions

Before we describe the decision tree algorithm, we need to define a few formulas.
 Let $S=A_1 \times A_2 ...\times A_k$ be a multiset of observations, $r$ a
"row" or "observation" in $S$, $A \in \lbrace A_1,...,A_k \rbrace$ an attribute set, and $r[A]$ a row in $A$.

Denote $|S|$ the number of observed
elements in $S$ (including repetition of the same element.)

We use the following definitions:

$$
    S_{A=j} = \lbrace r \in S  \lvert r[A] = j \rbrace 
$$

$$
    p(S, A, j) = \frac{\lvert S_{A=j} \rvert}{\lvert S \rvert}
$$

to describe the subset of the observations in $S$ that have value $j$
for attribute $A$ and the fraction of the observations in $S$
value $j$ for attribute $A$.

### Decision Tree Algorithm

Given a multiset of observations $S$, a target attribute
$T$ (that is, the label we are trying to predict), and a set,
$\text{ATTR}$, of possible attributes to split on, the basic algorithm
to build a decision tree, based on Hunt's algorithm, works as follows:

1. Create a tree node, $N$, with its class label set to the value from the target attribute $T$ that occurs most often:

$$
    \DeclareMathOperator*{\argmax}{argmax}
    \argmax\limits_{v \in \text{values}(T)} p(S, T, v)
$$

where $values(T)$ is the set of possible values for attribute $T$ and `argmax` yields the value $v$ that maximizes the function.  For interior nodes, the class label will be used when a traversal encounters an unexpected value for the split attribute.

2. If all the observations in $S$ are from the same target class, $\text{ATTR}$ is the empty set, or the remaining observations share the same values for the attributes in $\text{ATTR}$, return the node $N$.

3. Find the attribute $A$ from $\text{ATTR}$ that yields the largest gain ratio (defined below), set the split attribute for tree node $N$ to be $A$,and set the children of $N$ to be decision trees computed from the subsets obtained by splitting $S$ on $A$ with T as the target class and the remaining attributes ( $\text{ATTR} - \{A\}$ ) as the set of possible split attributes.  The edge from $N$ to the child computed from the subset $S_{A=j}$ should be labeled $j$.  Stop the recursion if the largest gain ratio is zero.

We use the term *gain* to describe the increase in purity with respect
to attribute $T$ that can be obtained by splitting the
observations in $S$ according to the value of attribute
$A$.  (In less formal terms, we want to identify the attribute
that will do the best job of splitting the data into groups such that
more of the members share the same value for the target attribute.)

There are multiple ways to define impurity, we'll use the gini
coefficient in this assignment:

$$
  \text{gini}(S, A) = 1 - \sum\limits_{j \in \text{values}(A)} p(S, A, j)^2
$$

Given that definition, we can define *gain* formally as:

$$
    \text{gain}(S, A, T) = \text{gini}(S, T) - \sum\limits_{j \in values(A)} p(S, A, j) * \text{gini}(S_{A=j}, T)
$$

We might see a large gain merely because splitting on an attribute
produces many small subsets.  To protect against this problem, we will
compute a ratio of the gain from splitting on an attribute to the
split information for that attribute:

$$
\text{gainratio}(S, A, T) = \frac{\text{gain}(S, A, T)}{\text{splitinfo}(S, A)}
$$

where split information is defined as:

$$
\text{splitinfo}(S, A) = - (\sum\limits_{j \in values(A)} p(S, A, j) * \log p(S, A, j))
$$

### Task 2: Building and using decision trees

We have seeded your directory with a file named
`decision_tree.py`.

This file includes a main block that processes
the expected command-line arguments--filenames for the training and testing data--and then calls a function named `go`.  
Your task is to implement `go` and any necessary auxiliary functions.
Your `go` function should build a decision tree from the training data and then return a list (or Pandas series) of the classifications obtained by using the decision tree to classify each observation in the testing data.

Your program must be able to handle any data set that:

1. has a header row,
2. has categorical attributes, and
3. in which the (binary) target attribute appears in the last column.  

You should use all the columns except the last one as attributes when building the decision tree.

You could break ties in steps 1 and 3 of the algorithm arbitrarily,
but to simplify the process of testing we will dictate a specific
method.  In step 1, choose the value that occurs earlier in the
natural ordering for strings, if both classes occur the same number of
times.  For example, if `"Yes"` occurs six times and `"No"` occurs
six times, choose `"No"`, because `"No" < "Yes"`.  In the unlikely
event that the gain ratio for two attributes `a1` and `a2`, where
`a1 < a2`, is the same, chose `a1`.

You *must* define a Python class to represent the nodes of the decision tree.
We strongly encourage you to use Pandas for this task as well.
It is well suited to the task of computing the different metrics (gini, gain, etc).

**Testing Task 2**

 We have provided test code for Task 2 in `test_decision_tree.py`.
