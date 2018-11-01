<meta name="keywords" content="numpy矩阵,numpy向量" />

# numpy中的矩阵和向量

The numpy ``ndarray`` class is used to represent both matrices and vectors. To construct a matrix in numpy we list the rows of the matrix in a list and pass that list to the numpy array constructor.

For example, to construct a numpy array that corresponds to the matrix

![矩阵1](/static/images/article/numpyLA1.png)

we would do

```python
A = np.array([[1,-1,2],[3,2,0]])
```

Vectors are just arrays with a single column. For example, to construct a vector

![矩阵2](/static/images/article/numpyLA2.png)

we would do

```python
v = np.array([[2],[1],[3]])
```

A more convenient approach is to transpose the corresponding row vector. For example, to make the vector above we could instead transpose the row vector

![矩阵3](/static/images/article/numpyLA3.png)

The code for this is

```python
v = np.transpose(np.array([[2,1,3]]))
```

numpy overloads the array index and slicing notations to access parts of a matrix. For example, to print the bottom right entry in the matrix A we would do

```python
print(A[1,2])
```

To slice out the second column in the A matrix we would do

```python
col = A[:,1:2]
```

The first slice selects all rows in A, while the second slice selects just the middle entry in each row.

To do a matrix multiplication or a matrix-vector multiplication we use the np.dot() method.

```python
w = np.dot(A,v)
```

## Solving systems of equations with numpy

One of the more common problems in linear algebra is solving a matrix-vector equation. Here is an example. We seek the vector x that solves the equation

<p class="eqn"><i>A</i> <b>x</b> = <b>b</b></p>

where

![矩阵4](/static/images/article/numpyLA4.png)

![矩阵5](/static/images/article/numpyLA5.png)

We start by constructing the arrays for A and b.

```python
A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
b = np.transpose(np.array([[-3,5,-2]])
```

To solve the system we do

```python
x = np.linalg.solve(A,b)
```

## Application: multiple linear regression

In a multiple regression problem we seek a function that can map input data points to outcome values. Each data point is a *feature vector (x1 , x2 , …, xm)* composed of two or more data values that capture various features of the input. To represent all of the input data along with the vector of output values we set up a input matrix X and an output vector **y**:

![矩阵6](/static/images/article/numpyLA6.png)

![矩阵7](/static/images/article/numpyLA7.png)

In a simple least-squares linear regression model we seek a vector <b>β</b> such that the product X β most closely approximates the outcome vector **y**.

Once we have constructed the <b>β</b> vector we can use it to map input data to a predicted outcomes. Given an input vector in the form

![矩阵8](/static/images/article/numpyLA8.png)

we can compute a predicted outcome value

![矩阵9](/static/images/article/numpyLA9.png)

The formula to compute the β vector is

<p class="eqn"><b>β</b> = (<i>X</i><sup><i>T</i></sup> <i>X</i>)<sup>-1</sup> <i>X</i><sup><i>T</i></sup> <b>y</b></p>

In our next example program I will use numpy to construct the appropriate matrices and vectors and solve for the <b>β</b> vector. Once we have solved for <b>β</b> we will use it to make predictions for some test data points that we initially left out of our input data set.

Assuming we have constructed the input matrix X and the outcomes vector **y** in numpy, the following code will compute the <b>β</b> vector:

```python
Xt = np.transpose(X)
XtX = np.dot(Xt,X)
Xty = np.dot(Xt,y)
beta = np.linalg.solve(XtX,Xty)
```

The last line uses ``np.linalg.solve`` to compute <b>β</b>, since the equation

<p class="eqn"><b>β</b> = (<i>X</i><sup><i>T</i></sup> <i>X</i>)<sup>-1</sup> <i>X</i><sup><i>T</i></sup> <b>y</b></p>

is mathematically equivalent to the system of equations

<p class="eqn">(<i>X</i><sup><i>T</i></sup> <i>X</i>) <b>β</b> = <i>X</i><sup><i>T</i></sup> <b>y</b></p>

The data set I will use for this example is the Windsor house price data set, which contains information about home sales in the Windsor, Ontario area. The input variables cover a range of factors that may potentially have an impact on house prices, such as lot size, number of bedrooms, and the presence of various amenities. A CSV file with the full data set is available here. I downloaded the data set from this site, which offers a large number of data sets covering a large range of topics.

Here now is the source code for the example program.

```python
import csv
import numpy as np

def readData():
    X = []
    y = []
    with open('Housing.csv') as f:
        rdr = csv.reader(f)
        # Skip the header row
        next(rdr)
        # Read X and y
        for line in rdr:
            xline = [1.0]
            for s in line[:-1]:
                xline.append(float(s))
            X.append(xline)
            y.append(float(line[-1]))
    return (X,y)

X0,y0 = readData()
# Convert all but the last 10 rows of the raw data to numpy arrays
d = len(X0)-10
X = np.array(X0[:d])
y = np.transpose(np.array([y0[:d]]))

# Compute beta
Xt = np.transpose(X)
XtX = np.dot(Xt,X)
Xty = np.dot(Xt,y)
beta = np.linalg.solve(XtX,Xty)
print(beta)

# Make predictions for the last 10 rows in the data set
for data,actual in zip(X0[d:],y0[d:]):
    x = np.array([data])
    prediction = np.dot(x,beta)
    print('prediction = '+str(prediction[0,0])+' actual = '+str(actual))
```
The original data set consists of over 500 entries. To test the accuracy of the predictions made by the linear regression model we use all but the last 10 data entries to build the regression model and compute <b>β</b>. Once we have constructed the <b>β</b> vector we use it to make predictions for the last 10 input values and then compare the predicted home prices against the actual home prices from the data set.

Here are the outputs produced by the program:

```python
[[ -4.14106096e+03]
 [  3.55197583e+00]
 [  1.66328263e+03]
 [  1.45465644e+04]
 [  6.77755381e+03]
 [  6.58750520e+03]
 [  4.44683380e+03]
 [  5.60834856e+03]
 [  1.27979572e+04]
 [  1.24091640e+04]
 [  4.19931185e+03]
 [  9.42215457e+03]]
prediction = 97360.6550969 actual = 82500.0
prediction = 71774.1659014 actual = 83000.0
prediction = 92359.0891976 actual = 84000.0
prediction = 77748.2742379 actual = 85000.0
prediction = 91015.5903066 actual = 85000.0
prediction = 97545.1179047 actual = 91500.0
prediction = 97360.6550969 actual = 94000.0
prediction = 106006.800756 actual = 103000.0
prediction = 92451.6931269 actual = 105000.0
prediction = 73458.2949381 actual = 105000.0
```

Overall, the predictions are not spectacularly good, but a number of the predictions fall somewhat close to being correct. Making better predictions from this data will be the subject of the winter term tutorial on machine learning.