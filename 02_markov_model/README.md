# CAPP 30122 - Programming Assignment #1 - Markovian Candidate

## Introduction

In this assignment, you will develop a modeling system using Markov Models.

Markov models can be used to capture the statistical relationships present in a language like English. These models allow us to go beyond simplistic observations, like the frequency with which specific letters or words appear in a language, and instead to capture the relationships between words or letters in sequences. As a result, we can not only appreciate that the letter “q” appears in text at a certain rate, but also that it is virtually always followed by a “u.” Similarly, “to be” is a much more likely word sequence than “to is.”

One application of Markov models is in analyzing actual text and assessing the likelihood that a particular person uttered it. That is one objective of this assignment.

The other objective for this assignment is to create a hash tables module. Hash tables are data structures that store associations between keys and values (exactly like dictionaries in Python) and provide an efficient means of looking up the value associated with a given key. Hash tables find a desired entry rapidly by limiting the set of places where it can be. They avoid “hot spots,” even with data that might otherwise all seem to belong in the same place, by dispersing the data through hashing.

Apart from developing an appreciation of how hashing and hash tables work, you will also be better prepared if ever you need to write your own hash table in the future. For instance, you may use a language (like C) that does not feature a built-in hash table. Or, the hash table that is used by a language, like Python, may interact poorly with your particular data, obligating you to design a custom-tailored one. After completing this assignment, you should consider hash tables to be in your programming repertoire.

## Part 0: Set Up the Project

Your first task is to set up the Speaker Recognition Project, as described in module #1.  We have provided the files & directories in the correct structure in your repository:

```
markovian/
  app.py
  collections/
    hashtable.py
    lp_hashtable.py
  models/
    markov.py
```

The top-level package is `markovian`, examples below will show you how to execute the project.

You will need to add the necessary file to ensure that running `python -m markovian` executes the `markovian.app.run` function.

There are also subpackages `collections`, and `models`.  You need to add the necessary files to make them subpackages of `markovian`.

As you implement the rest of the assignment, you may need to add `import` statements to import functions and classes from the appropriate submodules.

## Part 1: Hash Tables & Linear Probing

In the interest of building and testing code one component at a time, you will start by building a hash table. Once you have gotten this module up and running, you will use it in your construction of a Markov model for speaker attribution.

There are different types of hash tables; for this assignment, we will use the type that is implemented with **linear probing**.

Please look at `markovian/collections/lp_hashtable.py`. You will modify this file and must implement a hash table using the linear probing algorithm. The class `LPHashtable`:

* May assume all keys are strings.
* Must implement the a `_hash` function that takes in a string and returns a hash value.  Use the string hashing function discussed in class.
* Must define an `__init__(self, capacity, default_value)` constructor:
  * `capacity` - initial number of cells to use.  It must create a list of empty cells of the specified length. (You may assume this is always greater than zero.)
* It must use a **single** Python `list` (the already-defined `self._items`) to hold the key and value as a tuple in the table.  The tuples can contain any additional components as needed.
* Should define the other abstract methods defined in `hashtable.py`.
  * `__setitem__(key, val)` should assign (add or replace) the value associated with `key` to `val`.
  * `__getitem__(key)` should return the value associated with `key`, or the `default_val` if the key is not found.
  * `__len__` should return the total number of items currently stored within the table. (**Not capacity!**)
* You are free to implement additional properties, attributes, and methods as needed to implement this class.

### Hashing

You must use the hash function presented in these <http://www.cs.princeton.edu/courses/archive/spr03/cs226/lectures/hashing.4up.pdf> slides, and use 37 as a constant prime multiplier in that function. Use the ord() function in Python to compute the numerical value of a character. Let $f()$ be a function that maps a character $c_i$ in a string $S = c_1c_2...c_j$ to a number, $k=37$ be our polynomial constant, and $M$ the size of the hash table. One possible hash function (which you must use in this assignment) can be defined as (see slide #6):

$$h(S) = g(c_j)$$

where

$$
   g(c_i)=  
   \begin{cases}
   (g(c_{i-1})*k + f(c_{i})) \text{ mod }  M &, i \in \{2, 3, ...,j\}\\
   f(c_{i}) \text{ mod } M & , i=1
   \end{cases}
$$

### Rehashing

A hash table built on linear probing does not have unlimited capacity, since each cell can only contain a single value and there are a fixed number of cells.
But, we do not want to have to anticipate how many cells might be used for a given hash table in advance and hard-code that number in for the capacity of the hash table.
Therefore, we will take the following approach: the cell capacity passed into the `__init__` will represent the initial size of the table.
If the fraction of occupied cells grows beyond `TOO_FULL` (0.5) after an update, then we will perform an operation called rehashing:
We will expand the size of our hash table, and migrate all the data into their proper locations in the newly-expanded hash table (i.e., each key-value pair is hashed again, with the hash function now considering the new size of the table).
We will grow the size of the table by `GROWTH_RATIO`; since `GROWTH_RATIO = 2`, the size of the hash table will double each time it becomes too full.

## Part 2: A Speaker Recognition System

![](img/obama-mccain.png)

Markov models are used significantly in speech recognition systems, and used heavily in the domains of natural language processing, machine learning, and AI.

A Markov model defines a probabilistic mechanism for randomly generating sequences over some alphabet of symbols. A `k`-th order Markov model tracks the last `k` letters as the context for the present letter. We will build a module called `markov` that will work for any positive value of `k` provided.  This module will reside in `markovian/models/markov.py`

While we use the term "letter", we will actually work with all characters (letters, digits, spaces, punctuation). Our code will distinguish between upper & lower case letters.

### Building the Markov Model (Learning Algorithm)

Given a string of text from an unidentified speaker, we will use a Markov model for a known speaker to assess the likelihood that the text was uttered by that speaker. Thus, the first step is building a Markov Model on known text from a speaker. For this assignment, the Markov Model will be represented as a `LPHashtable`.

**You must use your implementation and not the built-in dictionaries from Python!**

You will be given an integer value of `k` at the start of the program. Each key-value pairing inside the table contains string keys with length `k` and `k + 1` and values set to the number of times those keys appeared in the text as substrings.

For example, let’s say you have a file called speakerA.txt that contains the following text:

```
This_is_.
```

**Note: You consider all characters in the text.**  Punctuation, special characters, whitespace characters, etc., are all considered valid.  Even if the text does not make any sense, you will still generate a Markov Model.

We will use this text `"This_is_."` to create a Markov model for Speaker A.  The algorithm is as follows:

Starting from the beginning of text from some known speaker:

1. For each character in the known text, you generate a string of length `k` that includes the current character plus k−1 succeeding characters (*Note:* The model actually works by finding the k preceding letters but our way works too because we are using a wrap-around effect.).
2. For each character in the known text, you generate a string of length `k+1` that includes the current character plus `k` succeeding characters.  (e.g. for the first character 'T', you'd generate 'Th' and 'Thi' if k=2)
3. For certain characters, they will not have `k` or `k+1` succeeding characters.   For example, what are the succeeding characters for the character `'.'`? if `k = 2` for our example text?  We will **wrap around**: we will think of the string circularly, and glue the beginning of the string on to the end to provide a source of the needed context.  (e.g. for the example string the strings generated for the last character '.' would be '.T' and '.Th'.)

Below is a diagram of all the k and k+1 length strings that are generated from the speakerA.txt file given k=2:

![](img/markov1.png)

The Markov model (i.e., your `LPHashtable`) will contain the number of times those `k` and `k + 1` were generated via the known text. Thus, for the speakerA.txt file, the Markov Model generated will be the following:

![](img/markov2.png)

Most of the `k` and `k+1` strings were only generated once, but some such as `"is"` occur more than once.

### Determining the likelihood of unidentified text (Testing algorithm)

As we stated earlier, given a string of text from an unidentified speaker, we will use the Markov model for a known speaker to assess the likelihood that the text was uttered by that speaker. Likelihood, in this context, is the probability of the model generating the unknown sequence.

**If we have built models for different speakers, then we will have likelihood values for each, and will choose the speaker with the highest likelihood as the probable source.**

These probabilities can be very small, since they take into account every possible phrase of a certain length that a speaker could have uttered. Therefore, we expect that all likelihoods are low in an absolute sense, but will still find their relative comparisons to be meaningful. Very small numbers are problematic, however, because they tax the precision available for floating-point values. The solution we will adopt for this problem is to use log probabilities instead of the probabilities themselves. This way, even a very small number is represented by a negative value in the range between zero and, for instance, -20. If our **log probabilities** are negative and we want to determine which probability is more likely, will the greater number (the one closer to zero) represent the higher or lower likelihood, based on how logarithms work?

Note that we will use the `math.log` function, which calculates natural logarithms. Your code should use this base for its logarithms. While any base would suffice to ameliorate our real number precision problem, we will be comparing your results to the results from our implementation, which itself uses natural logs.

The process of determining likelihood given a model is similar to the initial steps of building the Markov Model in the previous section. Starting from the beginning of text for some unknown speaker:

1. For each character in the unknown text, you generate a string of length k that includes the current character plus k − 1 succeeding characters.

2. For each character in the unknown text, you generate a string of length k + 1 that includes the current character plus k succeeding characters.

3. For certain characters, they will not have k or k + 1 succeeding characters. You will use the same wrap around mechanism as described previously.  (**Note:** These steps are identical to the steps in the learning process, think about what that suggests for your design!)

4. We then need to compute the probability that the string of length `k+1` follows a string of length `k`. To do this, look up in our data table the number of times that we have observed the string of length `k+1` (we will call this `M`), and the number of times we have observed the string of length `k` (we will call this `N`).  The log probability would be `log(M/N)`.

5. We need to keep in mind that we are constructing the model with one set of text and using it to evaluate other text. The specific letter sequences that appear in that new text are not necessarily guaranteed ever to have appeared in the original text. Consequently, we are at risk of dividing by zero.

It turns out that there is a theoretically-justifiable solution to this issue, called Laplace smoothing. We modify the simple equation above by adding to the denominator the number of unique characters that appeared in the original text we used for modeling. For instance, if every letter in the alphabet appeared in the text, we add 26. (In practice, the number is likely to be greater, because we distinguish between upper- and lower-case letters, and consider spaces, digits, and punctuation as well.) Because we have added this constant to the denominator, it will never be zero. Next, we must compensate for the fact that we have modified the denominator; a theoretically sound way to balance this is to add one to the numerator. Symbolically, if N is the number of times we have observed the k succeeding letters and M is the number of times we have observed those letters followed by the present letter, and S is the size of the ”alphabet” of possible characters, then our probability is

```
log((M + 1)/(N + S))
```

For example, lets say you have file called speakerC.txt (i.e., the unidentified text):

```
This
```

Calculating the total likelihood requires summing all of the individual likelihoods.  It will be done in the following way using the model we built from speakerA.txt:

![](img/markov3.png)

**Note:**

* When we say "Model(K)" we are looking at the hash table inside the model and retrieving the counts for that string.
* S is the number of unique characters that we encountered while building the model.  This means while building the model you need to create a set of all unique characters.

### Markov Class

Inside `markovian/models/markov.py`, you will implement a `Markov` class with the following methods

1. An `__init__(self, k, text)` method.

* `k` will be the order `k` for your Markov model.
* `text` will be the text used to build the model.

2. `log_probability(self, s)` takes a string and returns the log probability that the modeled speaker uttered it, using the approach described above.

### `identify_speaker`

You will also need to implement the function `identify_speaker(speech1, speech2, speech3, k)`.

This function will be called by the `main` function with three strings (`speech1`, `speech2`, `speech3`) and a value of `k` to instantiate your Markov model.

The function should learn models for `speech1` and `speech2`, calculate the **normalized log probabilities** that those two speakers uttered `speech3` and determine which speaker is more likely.

The return value of the function should be a 3-element tuple with the two calculated probabilities and a string "A" or "B" indicating which speaker most likely gave `speech3`.

While the log_probability function yields the likelihood that a particular speaker uttered a specified string of text, this probability may be misleading because it depends upon the length of the string. Because there are many more possible strings of a greater length, and these probabilities are calculated across the universe of all possible strings of the same length, we should expect to see significant variance in these values for different phrases. To reduce this effect, we will divide all of our probabilities by the length of the string, to yield normalized ones. To be clear, this division does not occur in log_probability, but rather in identify_speaker. Also, note that we will be calculating log probabilities, under different models, for the same string. Thus, string length differences are not a factor during a single run of our program. Rather, we choose to normalize in this fashion because we want outputs for different runs to be more directly comparable when they may pertain to different length quotes being analyzed.

**Note:** You can test your implementation of this file by running the tests inside `tests/test_markov.py`.  You must pass these tests in less than 60 seconds to avoid a penalty.

**Note:** There are additional test cases in `tests/test_markov_long.py`, these are mostly redundant with the tests in `test_markov` and can take a long time to run.

## Debugging Suggestions

* Check your code for handling the wrap-around carefully. It is a common source of errors.
* Test the code for constructing a Markov model using a very small string, such as "abcabd". Check your data structure to make sure that you have the right set of keys and the correct counts.
* Make sure you are using the correct value for S the number of unique characters that appeared in the training text.

## Acknowledgment

This assignment’s text and documentation originated from CAPP 30122 Team @ The University of Chicago. The original development of the assignment was done by Rob Schapire with contributions from Kevin Wayne.
