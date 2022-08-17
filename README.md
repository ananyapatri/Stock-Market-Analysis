# Stock-Market-Analysis

## Installing Dependencies

There are 3 important files. All of the files utilize the same dependencies which reside in environment.txt. First create a virtual environment using python3. In your terminal, create a folder, download all of this code in it, and enter this code:(make sure you are in the folder when you create the environment)

```
python3 -m venv senv
```

This creates a virtual environment with the name senv. To activate it, enter this code

```
source senv/bin/activate
```

now you are in the virtual environment. Next, enter this code to download the dependencies from requirments.txt

```
python -m pip install -r requirements.txt
```

## montecarlo.py

The Monte Carlo simulation is a model used to predict the probability of a variety of outcomes when the potential for random variables is present. This code runs scenarios based off of 2500 possible portfolios to select the best allocation of funds. These allocations are then changed using multiple signals to try and plot the best outcome. The main drawback is the time it takes to run.

## optimize.py

This model uses constrained optimization to achieve the same result and runs much faster.

## sigmoidal.py

This model uses the sigmoidal function to organize the signals and multiply the allocations to try and achieve a better result 
