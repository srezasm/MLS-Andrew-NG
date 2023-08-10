# Decision trees

## Decision tree model

- Each cell in the decision tree is called a _node_.
- The top most node is called the _root node_.
- The all other nodes, excluding the bottom most nodes are called _decision nodes_.
- The bottom most nodes are called _leaf nodes_.

<image src="./assets/img-10.jpg" height="300px">

Based on the count of features and possible value of each feature, there are several decision trees for each application. The job of a decision tree algorithm is to pick the one that does the best on the training set and also generalizes well to new data.

## Learning process

First off, we need to pick the root node feature.  
Then we should look at the left node and decide what feature to use next and so on till we pick the left most branch features.  
As for the left, we continue the process for the right branch and select the features for them as well.

1. How to choose what feature to split at each node? Select by the maximum purity(purity of a feature is the precision of that feature)
2. When do you stop splitting?
   - When a node is 100% one class.
   - When splitting a node will result in the tree exceeding a maximum depth(depth is the number of hops that takes to reach the selected node from root node)
     - Keeping the maximum depth small prevents the tree to become too big.
     - Keeping maximum depth small makes it less prone to overfitting.
   - When improvements in purity score are below a threshold.
   - When the number of examples in a node is below a threshold.

## Measuring purity

We use _entropy function_ to measure the impurity.

When $p_1$ is the fraction of examples that are positive, $H(p_1)$ is the entropy function of $p_1$.  
If $p_1 = \frac{3}{6}$, $H(p_1) = 1.0$, which tells it's _totally impure_.  
If $p_1 = \frac{5}{6}$, $H(p_1) = 0.65$.  
If $p_1 = \frac{6}{6}$, $H(p_1) = 0.0$, which tells it's _totally pure_.

<image src="./assets/img-11.jpg" height="300px">

We take $\log_2$, just to make the peak of the curve equal to one. If we where to take the $\log_e$, then that would vertically scale the current plot and would still work, but the peak of the function wouldn't be a round number anymore.

If $p_1 = 0$, we have $0\log(0)$, and the $\log(0)$ is undefined, but we consider the whole expression to be equal to 0 for convention.

_note)_ There's also another algorithm called _Gini_ that is used in decision trees, but we wont need it in this course.

## Choosing a split: Information gain

We'll decide what feature to split on at each node based on what choice of feature increases the purity the most.

The reduction of entropy is called _information gain_.

<image src="./assets/img-12.jpg" height="350px">

We get different entropies based on splitting by different features at each node. To find the best value, we take the sum of all the weighted averages of the entropies of each sub-node and subtract the result from the entropy of the current node to determine the improvement in purity.

### General _Information gain_ algorithm

$$
\begin{cases}
    IG(S, A) = E(S)- \displaystyle\sum_{v\ \in\ \text{Values}(A)} \dfrac{|S_v|}{|S|} E(S_v) \\
    E(S) = \displaystyle\sum_{i=1}^{n}-p(c_i)\log_2(p(c_i))
\end{cases}
$$

- $E(S)$: The current entropy on our node $S$, before any split
- $|S|$: The size or the number of instances in $S$
- $A$: An attribute in $S$ that has a given set of values (Let’s say it is a discrete attribute)
- $v$: Stands for value and represents each value of the attribute $A$
- $S_v$: After splitting $S$ using $A$, $S_v$ refers to each of the resulted subnodes from $S$, that share the same value in $A$
- $E(S_v)$: The entropy of a node $S_v$. This should be computed for each value of $A$ (assuming it is a discrete attribute)
- $p(c_i)$: The probability/percentage of class $c_i$ in a node  
  <small><a href="https://www.mldawn.com/the-decision-tree-algorithm-information-gain/">source</a></small>

## Putting it together

Overall process of making a decision tree:

1. Start with all the examples at the root node
2. Calculate the IG for all possible features, and pick the one with the highest IG
3. Split dataset according to the selected feature, and create left and right branches of the tree
4. Keep repeating the splitting process until the stopping criteria is met:
   - When a node is 100% one class
   - When spitting a node will result in the tree exceeding a maximum depth
   - IG from additional splits is less than threshold
   - When number of examples in a node is below a threshold

There are many ways to decide where to stop the tree, and one of the most common ways is to set a maximum depth.  
In theory we can use cross validation set to pick the parameter, although in practice, the open-source libraries have better ways to choose this parameter.  
Another way is to set a threshold for minimum IG, or number of items in each split.

## Using one-hot encoding of categorical features

If a categorical feature can take on $k$ values, create $k$ binary features. This can also be used for binary features too.

### Continuous valued features

When we are splitting, we'll consider different values to split on, carry out the usual IG calculation and decide to split on the value that gives the highest IG.

## Regression trees

In the case of regression, we can only predict the average of a group of examples that have been grouped together.

The key decision for this type of tree is to choose the features to split on.

When building a regression tree, instead of trying to reduce entropy, unlike the decision tree, we need to reduce the _variance_.

### _Variance Reduction_ algorithm

$$
\begin{cases}
    VR(S, A) = Var(S)- \displaystyle\sum_{v\ \in\ \text{Values}(A)} \dfrac{|S_v|}{|S|} Var(S_v) \\
    Var(S) = \displaystyle\sum_{i=1}^{n} \dfrac{(s_i - \mu)^2}{n}
\end{cases}
$$

- $Var(S)$: The current variance on our node $S$, before any split
- $|S|$: The size or the number of instances in $S$
- $A$: An attribute in $S$ that has a given set of values (Let’s say it is a discrete attribute)
- $v$: Stands for value and represents each value of the attribute $A$
- $S_v$: After splitting $S$ using $A$, $S_v$ refers to each of the resulted subnodes from $S$, that share the same value in $A$
- $Var(S_v)$: The variance of a node $S_v$. This should be computed for each value of $A$ (assuming it is a discrete attribute)
- $\mu$: Normal average of present values in given $S$

## Using multiple decision trees

A single decision trees can be highly sensitive to small changes in the data, and a solution to make it less sensitive is to make a lot of decision trees which is called a _tree ensemble_.

For example, it we change just one example in the set, the resulting decision tree might have a complete different structure.

## Sampling with replacement

Sampling with replacement is randomly picking an example from the dataset. We might get the same example more than once, but it's ok.

## Random forest algorithm

Given a training set of size $m$

For $b = 1$ to $B$:

- Use sampling with replacement to create a new training set of size $m$

We can choose $B$ anywhere from 64 to 228, and make a decision tree per each resulting set, choosing any large $B$ would never hurt performance and accuracy, but beyond a certain point it would make the application slow and wouldn't make the accuracy much better. The recommended $B$ is abut 100.

After we have all the random decision trees, we can give the new examples to all of the trees and select the most repeated result as the final result.

This specific instance creation of tree ensemble is sometimes called _bagged decision tree_. The term $B$ actually comes from _Bagged_ too.

A change will make it a _random forest_; The problem with this bagged decision tree is, even though with this sampling with replacement algorithm, sometimes the generated trees will be similar in the splits and root node.

### Randomizing the feature choice

At each node, when choosing a feature to use to split, if $n$ features are available, pick a random subset of $k < n$ features and allow the algorithm to only choose from that subset of features.

When $n$ is large, a typical choice of value $k$ would be $k = \sqrt{n}$

Now by this change, we have the _random forest_ algorithm.

<details>
    <summary>Awful joke</summary>
    Where do a machine learning engineer go camping?<br>
    Random Forest.
    <details>
        <summary>makeup information</summary>
        <la>
            <li><a href="https://huggingface.co">Huggingface 🤗: Leading platform to share machine learning models and datasets.</a></li>
            <li><a href="https://kaggle.com">Kaggle: ML competitions, datasets and share and collaborate on code and ideas</a></li>
            <li><a href="https://github.com">Github: GITHUB!</a></li>
        </la>
    </details>
</details>

## XGBoost

The most popular decision tree algorithm is XGBoost.

The idea of boosting comes from in each iteration, increasing the probability of choosing the mispredicted classes which changes the algorithm into:

For $b = 1$ to $B$:

- Use sampling with replacement to create a new training set of size $m$. But instead of picking from all examples with equal (1/m) probability, make it more likely to pick misclassified examples from previously trained trees.
- Train a decision tree on the new dataset.

### XGBoost (EXtreme Gradient Boosting)

- Open source implementation of boosted trees
- Fast efficient implementation
- Good choice of default splitting criteria and criteria for when to stop splitting
- Built in regularization to prevent overfitting
- Highly competitive algorithm for machine learning competitions e.g. Kaggle competitions

_note)_ Rather than doing something with replacement, XGBoost assigns different weights to different training examples; so it doesn't need to generate a lot of randomly chosen training sets, and this way is more efficient than sampling and replacement.

_Classification_ using sample:

```python
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

_Regression_ using sample:

```python
from xgboost import XGBRegressor

model = XGBRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## When to use decision trees

Decision Trees and Tree ensembles

- Works well on tabular (structured) data
- Not recommended for unstructured data(images, audio, text)
- It's very fast to train and predict
- Small decision trees may be human interpretable

Neural Networks

- Works well on all types of daa, including tabular (structured) and unstructured data
- May be slower than a decision tree
- Works with transfer learning
- When building a system of multiple models working together, it might be easier to string together multiple neural networks
