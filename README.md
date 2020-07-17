
# Prologue

For this project we will use the logistic regression function to model the growth of confirmed Covid-19 case population growth in Bangladesh. The logistic regression function is commonly used in classification problems, and in this project we will be examining how it fares as a regression tool. Both cumulative case counts over time and logistic regression curves have a sigmoid shape and we shall try to fit a theoretically predicted curve over the actual cumulative case counts over time to reach certain conclusions about the case count growth, such as the time of peak daily new cases and the total cases that may be reached during this outbreak.

# Import the necessary modules


```python
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
%matplotlib inline
```

# Connect to Google Drive (where the data is kept)


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    

# Import data and format as needed


```python
df = pd.read_csv('/content/drive/My Drive/Corona-Cases.n-1.csv')
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Total cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>127</th>
      <td>7-13-2020</td>
      <td>186894</td>
    </tr>
    <tr>
      <th>128</th>
      <td>7-14-2020</td>
      <td>190057</td>
    </tr>
    <tr>
      <th>129</th>
      <td>7-15-2020</td>
      <td>193590</td>
    </tr>
    <tr>
      <th>130</th>
      <td>7-16-2020</td>
      <td>196323</td>
    </tr>
    <tr>
      <th>131</th>
      <td>7-17-2020</td>
      <td>199357</td>
    </tr>
  </tbody>
</table>
</div>



As you can see, the format of the date is 'month-day-year'. Let's specify the date column is datetime type. Let's also specify the formatting as %m-%d-%Y. And then, let's find the day when the first confirmed cases of Covid-19 were reported in Bangladesh.


```python
FMT = '%m-%d-%Y'
df['Date'] = pd.to_datetime(df['Date'], format=FMT)
```

We have to initialize the first date of confirmed Covid-19 cases as the datetime variable start_date because we would need it later to calculate the peak.


```python
# Initialize the start date
start_date = datetime.date(df.loc[0, 'Date'])
print('Start date: ', start_date)
```

    Start date:  2020-03-08
    

Now, for the logistic regression function, we would need a timestep column instead of a date column in the dataframe. So we create a new dataframe called data where we drop the date column and use the index as the timestep column.


```python
# drop date column
data = df['Total cases']

# reset index and create a timestep
data = data.reset_index(drop=False)

# rename columns
data.columns = ['Timestep', 'Total Cases']

# check
data.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestep</th>
      <th>Total Cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>127</th>
      <td>127</td>
      <td>186894</td>
    </tr>
    <tr>
      <th>128</th>
      <td>128</td>
      <td>190057</td>
    </tr>
    <tr>
      <th>129</th>
      <td>129</td>
      <td>193590</td>
    </tr>
    <tr>
      <th>130</th>
      <td>130</td>
      <td>196323</td>
    </tr>
    <tr>
      <th>131</th>
      <td>131</td>
      <td>199357</td>
    </tr>
  </tbody>
</table>
</div>



# Defining the logistic regression function


```python
def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))
```

In this formula, we have the variable x that is the time and three parameters: a, b, c.
* a is a metric for the speed of infections
* b is the day with the estimated maximum growth rate of confirmed Covid-19 cases
* c is the maximum number the cumulative confirmed cases will reach by the end of the first outbreak here in Bangladesh

The growth of cumulative cases follows a sigmoid shape like the logistic regression curve and hence, this may be a good way to model the growth of the confirmed Covid-19 case population over time. For the first outbreak at least. It makes sense because, for an outbreak, the rise in cumulative case counts is initially exponential. Then there is a point of inflection where the curve nearly becomes linear. We assume that this point of inflection is the time around which the daily new case numbers will peak. After that the curve eventually flattens out. 



# Fit the logistic function and extrapolate


```python
# Initialize all the timesteps as x
x = list(data.iloc[:,0])

# Initialize all the Total Cases values as y
y = list(data.iloc[:,1])

# Fit the curve using sklearn's curve_fit method we initialize the parameter p0 with arbitrary values
fit = curve_fit(logistic_model,x,y,p0=[2,100,20000])
(a, b, c), cov = fit
```


```python
# Print outputs
print('Metric for speed of infections: ', a)
print('Days from start when cumulative case counts will peak: ', b)
print('Total cumulative cases that will be reached: ', c)
```

    Metric for speed of infections:  16.261895219019074
    Days from start when cumulative case counts will peak:  107.33718062635585
    Total cumulative cases that will be reached:  242843.04639239082
    


```python
# Print errors for a, b, c
errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]
print('Errors in a, b and c respectively:\n', errors)
```

    Errors in a, b and c respectively:
     [0.09587726005485112, 0.2269164109726332, 1398.2032327752302]
    


```python
# estimated time of peak
print('Estimated time of peak between', start_date + timedelta(days=(b-errors[1])), ' and ', start_date + timedelta(days=(b+errors[1])))

# estimated total number of infections 
print('Estimated total number of infections betweeen ', (c - errors[2]), ' and ', (c + errors[2]))
```

    Estimated time of peak between 2020-06-23  and  2020-06-23
    Estimated total number of infections betweeen  241444.8431596156  and  244241.24962516606
    

To extrapolate the curve to the future, use the fsolve function from scipy.


```python
# Extrapolate
sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))
```

# Plot the graph


```python
pred_x = list(range(max(x),sol))
plt.rcParams['figure.figsize'] = [7, 7]
plt.rc('font', size=14)
# Real data
plt.scatter(x,y,label="Real data",color="red")
# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], label="Logistic model" )
plt.legend()
plt.xlabel("Days since 8th March 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))
plt.show()
```


![png](output_25_0.png)


# Evaluate the MSE error

Evaluating the mean squared error (MSE) is not very meaningful on its own until we can compare it with another predictive method. We can compare MSE of our regression with MSE from another method to check if our logistic regression model works better than the other predictive model. The model with the lower MSE performs better.





```python
y_pred_logistic = [logistic_model(i,fit[0][0],fit[0][1],fit[0][2])
for i in x]

print('Mean squared error: ', mean_squared_error(y,y_pred_logistic))
```

    Mean squared error:  1040576.8324953101
    

# Epilogue

We should be mindful of some caveats:

* These predictions will only be meaningful when the peak has actually been crossed definitively. 

* Also, the reliability of the reported cases would also influence the dependability of the model. Developing countries, especially the South Asian countries have famously failed to report accurate disaster statisticcs in the past. 

* Also, the testing numbers are low overall, especially in cities outside Dhaka where the daily new cases still have not peaked yet.

* Since most of the cases reported were in Dhaka, the findings indicate that the peak in Dhaka may have been reached already.

* If there is a second outbreak before the first outbreak subsides, the curve may not be sigmoid shaped and hence the results may not be as meaningful.

* The total reported case numbers will possibly be greater than 238000, because the daily new cases is still rising in some cities other than Dhaka. It is not unsound to expect that the total reported case count for this first instance of Covid-19 outbreak could very well reach 300000 or more.

* The government recently hiked the prices of tests which may have led to increased unwillingness in suspected candidates to actually test for the disease, and that may have influenced the recent confirmed case counts.

# References

Inspiration for theory and code from the following articles:

* [Covid-19 infection in Italy. Mathematical models and predictions](https://towardsdatascience.com/covid-19-infection-in-italy-mathematical-models-and-predictions-7784b4d7dd8d)

* [Logistic growth modelling of COVID-19 proliferation in China and its international implications](https://www.sciencedirect.com/science/article/pii/S1201971220303039)

* [Logistic Growth Model for COVID-19](https://www.wolframcloud.com/obj/covid-19/Published/Logistic-Growth-Model-for-COVID-19.nb)

