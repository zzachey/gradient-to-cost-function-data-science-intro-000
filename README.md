
### Introduction

In the previous lesson, we learned the mathematical defintion of a gradient.  We saw that the gradient of a function was a combination of our partial derivatives with respect to each variable of that function.  We saw the direction of gradient descent was simply to move in the negative direction of the gradient.  For example, if the direction of ascent of a function is a move up and to the right, the descent is down and to the left. In this lesson we will apply gradient descent to our cost function to see how we can move towards a best fit regression line by changing variables of $m$ and $b$.  

### Representing RSS as a multivariable function

Think about why gradient descent applies so well to a cost function.  Initially, we said that the cost of our function, meaning the difference between what our regression line predicted and the dataset, changed as we altered the y-intercept or the slope of the function.

Remember that mathematically, when we say cost function, we use the residual sum of squares where $$ RSS = \sum_{i=1}^n(actual - expected)^2 = \sum_{i=1}^n(y_i - \overline{y})^2 = \sum_{i=1}^n(y_i - mx_i + b)^2$$ for all $x$ and $y$ values of our dataset.  So in the graph directly below, $x_i$ and $y_i$  would be our points representing a movie's budget and revenue.  Meanwhile, $mx_i + b $ is our predicted $y$ value for a given $x$ value, of a budget. 

And RSS takes the difference between $mx_i + b$, the $y_i$ value our regression line predicts, and our actual $y$, represented by the length of the red lines.  Then we square this difference, and sum up these squares for each piece of data in our dataset.  That is the residual sum of squares.

![regression-scatter.png](./regression-scatter.png)

And we when we just plotted how RSS changes as we change one variable of our regression line, $m$ or $b$, we note how this looks like a curve, and call it our cost curve.


```python
import plotly
from plotly.offline import init_notebook_mode, iplot
from graph import m_b_trace, trace_values, plot
init_notebook_mode(connected=True)
b_values = list(range(70, 150, 10))
rss = [10852, 9690, 9128, 9166, 9804, 11042, 12880, 15318]
cost_curve_trace = trace_values(b_values, rss, mode="line", name = 'RSS with changes to y-intercept')
plot([cost_curve_trace])
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



<div id="8180eb52-f077-432f-94c4-0c0c08f25f9d" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("8180eb52-f077-432f-94c4-0c0c08f25f9d", [{"x": [70, 80, 90, 100, 110, 120, 130, 140], "y": [10852, 9690, 9128, 9166, 9804, 11042, 12880, 15318], "mode": "line", "name": "RSS with changes to y-intercept", "text": []}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


In two dimensions, we decrease our RSS simply by moving forwards or backwards along the cost curve which is the equivalent of changing our variable, in this case y-intercept.  So the cost curve above indicates that changing the regression line from having a y-intercept of 70 to 80 decreases our cost, the RSS.

Allowing us to change both variables, $m$ and $b$ means calculating how RSS varies with both $m$ and $b$. 

Because the RSS is a function of how we change our values of $m$ and $b$, we can express this relationship mathematically by saying the cost function, $J$ is the following:  
$$J(m, b) = \sum_{i=1}^{n}(y_i - (mx_i + b))^2$$

In the function above, $J$ is a function of $m$ and $b$. $J$ just represents the residual sum of squares, which varies as the $m$ and $b$ variables of our regession line are changed.  

Just our other multivariable functions we have seen thus far, we can display it in three dimensions, and it looks like the following.

![](./gradientdescent.png)

The three-dimensional graph above shows how the cost associated with our regression line changes as the slope and y-intercept values are changed.  

### Calculating the gradient of our cost function

Let's explore using gradient descent to determine how to change our regression line when we can alter both $m$ and $b$ variables.  When applied to a general multivariable function $f(x,y)$, gradient descent answered how much move the $x$ variable and the $y$ variable to produce the greatest decrease in output.  

> Now that we are applying gradient descent to our cost curve $J(m, b)$, the technique should answer how much to move the $m$ variable and the $b$ variable to produce the greatest decrease in cost, or RSS. In other words, when altering our regression line, we want to know how much of this change should be derived from a move in the slope versus how much should be derived from a change in the y-intercept.

As we know, the gradient of a function is simply the partial derivatives with respect to each of the variables, so:

$$ \nabla J(m, b) = \frac{\delta J}{\delta m}, \frac{\delta J}{\delta b}$$

In calculating the partial derivatives of our function $J(m, b) = \sum_{i=1}^{n}(y_i - (mx_i + b))^2$, we won't change the result if we ignore the summation until the very end.  We'll do that to make our calculations easier.

Ok, so let's take our partial derivatives of the following:

$$\frac{\delta J}{\delta m}J(m, b) = \frac{\delta J}{\delta m}(y - (mx + b))^2$$

$$\frac{\delta J}{\delta b}J(m, b) = \frac{\delta J}{\delta b}(y - (mx + b))^2$$

### Taking our first partial derivative

Let's start with taking the partial derivative with respect to $m$.

$$\frac{\delta J}{\delta m}J(m, b) = \frac{\delta J}{\delta m}(y - (mx + b))^2$$

Now this is a tricky function to take the derivative of.  So we can use functional composition followed by the chain rule to make it easier.  Using functional composition, we can rewrite our function $J$ as two functions: 

$$g(m,b) = y - (mx + b)$$

$$J(g(m,b)) = (g(m,b))^2$$

Now using the chain rule to find the partial derivative with respect to a change in the slope, gives us:

$$\frac{dJ}{dm}J(g) = \frac{dJ}{dg}J(g(m, b))*\frac{dg}{dm}g(m,b)$$

Our next step is to solve these derivatives individually: 

$$\frac{dJ}{dg}J(g(m, b)) = \frac{dJ}{dg}g(m,b)^2 = 2*g(m,b)$$

$$\frac{dg}{dm}g(m,b) =  \frac{dg}{dm} (y - (mx +b)) = \frac{dg}{dm}y - \frac{dg}{dm}mx - \frac{dg}{dm}b = -x $$

> Each of the terms are treated as constants, except for the middle term.  

Now plugging these back into our chain rule we have: 

 $\frac{dJ}{dg}J(g(m,b))*\frac{dg}{dm}g(m,b) = (2*g(m,b))*-x = 2*(y - (mx + b))*-x $
 
 So
 
$$\frac{\delta J}{\delta m}J(m, b) =  2*(y - (mx + b))*-x = -2x*(y - (mx + b ))  $$

### Our second partial derivative

Ok, now let's calculate the partial derivative with respect to a change in the y-intercept.  We express this mathematically with the following:

$$\frac{\delta J}{\delta b}J(m, b) = \frac{dJ}{db}(mx + b - y)^2$$

Then once again, we use functional composition following by the chain rule.  So we view our cost function as the same two functions $g(m,b)$ and $J(g(m,b))$.  

$$g(m,b) = y - (mx + b)$$

$$J(g(m,b)) = (g(m,b))^2$$

So applying the chain rule, to this same function composition, we get:

$$\frac{dJ}{db}J(g) = \frac{dJ}{dg}J(g)*\frac{dg}{db}g(m,b)$$

Now, our next step is to calculate these partial derivatives individually.

From our earlier calculation of the partial derivative, we know that $\frac{dJ}{dg}J(g(m,b)) = \frac{dJ}{dg}g(m,b)^2 = 2*g(m,b)$.  The only thing left to calculate is $\frac{dg}{db}g(m,b)$.

$\frac{dg}{db}g(m,b) = \frac{dg}{db}(y - (mx + b) ) = -1$

Now we plug our terms into our chain rule and get: 

$$ \frac{dJ}{dg}J(g)*\frac{dg}{db}g(m,b) = 2*g(m,b)*-1 = -2*(y - (mx + b)) $$

### Using both of our partial derivatives for gradient descent

Ok, so now we have our two partial derivatives for $\nabla J(m, b)$:

$$ \frac{dJ}{dm}J(m,b) = -2*x(y - (mx + b )) $$
$$ \frac{dJ}{db}J(m,b) = -2*(y - (mx + b)) $$

And as $mx + b$ = is just our regression line, we can simplify these formulas to be: 

$$ \frac{dJ}{dm}J(m,b) = -2*x(y - \overline{y})  = -2x*\epsilon$$
$$ \frac{dJ}{db}J(m,b) = -2*(y - \overline{y}) = -2\epsilon$$

Remember, `error` = `actual` - `expected`, so we can replace $y - \overline{y}$ with $\epsilon$, our error. As we mentioned above, our last step is adding back the summations.  Since $-2$ is a constant, we can keep this outside of the summation.  Our value for $x$ changes depending upon what x value we are at, so it must be included inside the summation for the first equation.  Below, we have: 

$$ \frac{dJ}{dm}J(m,b) = -2*\sum_{i=1}^n x(y_i - \overline{y}_i)  = -2*\sum_{i=1}^n x_i*\epsilon_i$$
$$ \frac{dJ}{db}J(m,b) = -2*\sum_{i=1}^n(y_i - \overline{y}_i) = -2*\sum_{i=1}^n \epsilon_i$$

So that is what what we'll do to find the "best fit regression line."  We'll start with an initial regression line with values of $m$ and $b$.  Then we'll go through our dataset, and we will use the above formulas with each point to tell us how to update our regression line such that it continues to minimize our cost function.

In the context of gradient descent, we use these partial derivatives to take a step size.  Remember that our step should be in the opposite direction of our partial derivatives as we are *descending* towards the minimum.  So to take a step towards gradient descent we use the general formula of:

`current_m` =  `old_m` $ - \frac{dJ}{dm}J(m,b)$

`current_b` =  `old_b` $ - \frac{dJ}{db}J(m,b) $

or in the code that we just calculated:

`current_m` = `old_m` $ -  (-2*\sum_{i=1}^n x_i*\epsilon_i )$

`current_b` =  `old_b` $ - ( -2*\sum_{i=1}^n \epsilon_i )$

In the next lesson, we'll work through translating this technique, with use of our $\nabla J(m, b)$, into code to descend along our cost curve and find the "best fit" regression line.

![regression-scatter.png](./regression-scatter.png)

### Summary

In this section, we developed some intuition for why the gradient of a function is the direction of steepest ascent and the negative gradient of a function is the direction of steepest decent.  Essentially, the gradient uses the partial derivatives to see what change will result from changes in the function's dimensions, and then moves in that direction weighted more towards the partial derivative with the larger magnitude.

We also practiced calculating some gradients, and ultimately calculated the gradient for our cost function.  This gave us two formulas which tell us how to update our regression line so that it descends along our cost function and approaches a "best fit line".
