##NOTE
To run the web app locally, please type the following commoand after installing the libraries from requirements.txt
streamlit run allaboutcars.py 

# AllBoutCars

AllBoutCars is a web-application made for Microsoft Engage 2022. The theme for the project was to create a web-app using Data Analysis in the Automotive Sector.
In this project I have used 3 databases- the sample dataset given in the engage program, electric car dataset as well as a used car dataset.

I have mainnly implemented three features which are explained below 

1) The first feature is a comprehensive analysis of Conventional cars i.e petrol/diesel/CNG vs Electric Cars. The user is prompted to enter the details such as make,model,mileage,fuel type etc and the running cost per km and the total cost is displayed for both the car types.
![image](https://user-images.githubusercontent.com/78296720/170887965-1a0ae142-8878-41d0-b47e-cb5256616980.png)
![image](https://user-images.githubusercontent.com/78296720/170887977-ebbf911e-79a9-463f-8ee7-0a827d21e7a9.png)
The above images shows the results displayed after all the input variables are entered by the user.

![image](https://user-images.githubusercontent.com/78296720/170888043-aebb3530-5420-4802-99a0-2b313a175bc4.png)
The above graph shows the ex showroom price of the conventional car type vs the electric car type. A trend is seen by this graph that the price of the electric car is a lot higher as compared to the conventional car type.

The second graph shows us the fuel expenditure which is amount spent on petrol/diesel by the conventional cars vs the electricty cost spend while charging the electric vehicle. Here, on the other hand the fuel expenditure for the conventional car type is higher as compared to the electric car.



2) The second feature lets the user put filters based on various parameters such as make,fuel type,price and mileage and a table with corresponding models of cars are displayed.
![image](https://user-images.githubusercontent.com/78296720/170888212-df999f13-8097-433f-b7d3-3f513e20cc56.png)
On clicking view more details a range of other features can be viewed.


3) The third feature can predict the selling price for a used car. By taking various parameters like make,model,fuel type, years it will predict the range of the price for which the car can be sold. Here for making the model linear regression model was used. 
![image](https://user-images.githubusercontent.com/78296720/170888299-9829c825-ad62-46f5-b01c-22ed5cee32bb.png)
