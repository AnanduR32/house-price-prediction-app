# House price prediction app
A simple house price prediction application written in python trained using sklearn Gradient boosted Linear Regression model, build using Dash framework from plotly and deployed on Heroku

### app deployment : 
https://house-price-ar32.herokuapp.com/

In certain districts the number of bathrooms, living rooms, drawing rooms and/or kitchens are limited to 1 or two, therefore the slider's max value is locked based on this information.

Number of floors will not be considered for the modelling at present since some people could be living on higher floors apartments with many floors, and some in houses that has 1 or 2 floor, this could possibly throw off the model unless handled accordingly. 

By taking the median value of each of the columns:  
- square  
- livingRoom  
- drawingRoom  
- kitchen  
- bathRoom  
- buildingType  
- renovationCondition  
- buildingStructure  
- elevator  
- fiveYearsProperty  
- subway  
- district  

We can derive the metric - 'popularity' based on the values selected which is calculated by measuring how far away from the 50th percentile data the selected data is and then calculating the ordinary least squares error - through which the percentage 'popularity'

Obtained beijing topographical map from Nicole9519's [repo](https://github.com/Nicole9519/Nicole9519.github.io/blob/54284226e0266763611a51ee925f9b7a9f80c5e2/dist/data/beijing.geojson) which had to be formatted to be used in the dash board, the formatting/cleaning can be found [here](https://github.com/AnanduR32/MachineLearning-Basics/tree/master/Case%20Studies/Python/Beijing%20House%20price%20prediction)

Edited the geojson map using geopandas and http://geojson.io/
