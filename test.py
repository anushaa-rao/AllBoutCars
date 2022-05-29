from asyncore import write
from gettext import npgettext
from msilib.schema import CheckBox

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import pickle
import plotly.express as px
import altair as alt
#import plotly.graph_objects as go
import numpy as np
st.set_page_config(layout="wide")
st.title("Automotive Industry Data Analysis")
selected = option_menu(
            menu_title=None,  
            options=["Suggest Cars", "Predict Price of Used Car","Electric vs Conventional Car"], 
            default_index=0, 
            orientation="horizontal",
            
        )
            
        


if selected == "Suggest Cars":
    #df = pd.read_csv(r"C:\Users\Anusha\Desktop\project\cars_engage_2022.csv")
    df = pd.read_csv(r"https://github.com/anushaa-rao/Check/blob/main/cars_engage_2022.csv")
    

    #cleaning data

    
    df['City_Mileage']=df["City_Mileage"].str.replace("?","")
    df['City_Mileage']=df["City_Mileage"].str.replace(" km/litre","")
    df['City_Mileage']=df["City_Mileage"].str.replace(",",".")
    df['Ex-Showroom_Price']=df["Ex-Showroom_Price"].str.replace("Rs. ","")
    df['Ex-Showroom_Price']=df["Ex-Showroom_Price"].str.replace(".","")
    df['Ex-Showroom_Price']=df["Ex-Showroom_Price"].str.replace(",","")
        #make filter input
    Make = df['Make'].unique()
    make_choice=st.sidebar.selectbox('Select preferred Make:', Make)

    df['City_Mileage'] = df['City_Mileage'].astype(float)
    #options for inputs from user
    
    Fuel_Type = df['Fuel_Type'].unique()
   
    
    #City_Mileage = df['City_Mileage'].unique()
    
    #fuel filter input
    fuel_choice = st.sidebar.selectbox('Select preferred type of Fuel:',Fuel_Type)

    #Mileage filter
    df['City_Mileage'] = df['City_Mileage'].astype(float)
    mileage_choice = st.sidebar.slider('Select preferred mileage range',0,50, (5, 15))
    df_mileage= df['City_Mileage'].between(mileage_choice[0], mileage_choice[1])
    
    #Ex-Showroom_Price filter
    df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].astype(float)
    price_choice = st.sidebar.slider('Select a price range',0,7000000, (0, 2500000))
    df_price= df['Ex-Showroom_Price'].between(price_choice[0], price_choice[1])
    


    
    #filtering data based on inputs 
    filter1 = df["Make"].isin([make_choice])
    filter2 = df["Fuel_Type"].isin([fuel_choice])
    filter3 = df_mileage
    filter4 = df_price
    
    
    # displaying data with filters applied
    df=df[filter1 & filter2 & filter3 & filter4]
   
    df.drop(df.columns.difference(['Make','Ex-Showroom_Price','Fuel_Type','City_Mileage']), 1, inplace=True)
    
    if df.empty:
        st.warning("No cars available with the selected features")
    else:
        st.table(df)

    


if selected == "Predict Price of Used Car":
      
    model = pickle.load(open('RF_price_predicting_model.pkl','rb'))


    def main():
        st.title("Selling Price Predictor")
        st.markdown("Are you planning to sell your car?\n##### Enter the following details to get the predicted price for your car")

        # @st.cache(allow_output_mutation=True)
        # def get_model():
        #     model = pickle.load(open('RF_price_predicting_model.pkl','rb'))
        #     return model

        st.write('')
        st.write('')
        #taking inputs of various parameters 
        years = st.number_input('Which year was the car purchased?',1990, 2020, step=1, key ='year')
        Years_old = 2020-years

        Present_Price = st.number_input('Current ex-showroom price of the car (In lakhs)', 0.00, 50.00, step=0.5, key ='present_price') 

        Kms_Driven = st.number_input('Distance completed by the car in Kilometers', 0.00, 500000.00, step=500.00, key ='drived')

        Owner = st.radio("Number of owners the car had previously", (0, 1, 3), key='owner')


        Fuel_Type_Petrol = st.selectbox('Fuel type of the car',('Petrol','Diesel', 'CNG'), key='fuel')
        if(Fuel_Type_Petrol=='Petrol'):
            Fuel_Type_Petrol=1
            Fuel_Type_Diesel=0
        elif(Fuel_Type_Petrol=='Diesel'):
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=1
        else:
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=0

        Seller_Type_Individual = st.selectbox('Dealer or Individual', ('Dealer','Individual'), key='dealer')
        if(Seller_Type_Individual=='Individual'):
            Seller_Type_Individual=1
        else:
            Seller_Type_Individual=0	

        Transmission_Mannual = st.selectbox('Transmission Type', ('Manual','Automatic'), key='manual')
        if(Transmission_Mannual=='Mannual'):
            Transmission_Mannual=1
        else:
            Transmission_Mannual=0


        if st.button("Estimate Price", key='predict'):
            try:
                Model = model  #get_model()
                prediction = Model.predict([[Present_Price, Kms_Driven, Owner, Years_old, Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Mannual]])
                output = round(prediction[0],2)
                if output<0:
                    st.warning("You will be not able to sell this car!")
                else:
                    st.success("Estimated selling price for the car is {} lakhs".format(output))
            except:
                st.warning("Something went wrong\nTry again")
                



    if __name__ == "__main__":
        main()


if selected == "Electric vs Conventional Car":
    #petrol vs electric car analysis
    st.title('Cost Benefit Analysis')

    col1, col2 = st.columns([3, 3])

    col1.subheader("Select Conventional Car")
    col2.subheader("Select Electric Car")

    #reading normal car and electric car dataset
    #nor_com=pd.read_csv(r"C:\Users\Anusha\Desktop\project\cars_engage_2022.csv")
    nor_com=pd.read_csv(r"https://github.com/anushaa-rao/Check/blob/main/cars_engage_2022.csv")
    #ev_com=pd.read_csv(r"C:\Users\Anusha\Desktop\project\ElectricCar.csv")
    ev_com=pd.read_csv(r"https://github.com/anushaa-rao/Check/blob/main/ElectricCar.csv")

    with col1:
        make_options = nor_com['Make'].unique().tolist()
        ev_options = ev_com['Make'].unique().tolist()

        #input and filtering based on make 
        sel_make=st.selectbox('Manufacturer',make_options)
        new_nor=nor_com.loc[nor_com['Make'] == sel_make]

        #cleaning dataframe obtained 
        new_nor['Ex-Showroom_Price']=new_nor["Ex-Showroom_Price"].str.replace("Rs. ","")
        new_nor['Ex-Showroom_Price']=new_nor["Ex-Showroom_Price"].str.replace(".","")
        new_nor['Ex-Showroom_Price']=new_nor["Ex-Showroom_Price"].str.replace(",","")

        new_nor['City_Mileage']=new_nor["City_Mileage"].str.replace("?","")
        new_nor['City_Mileage']=new_nor["City_Mileage"].str.replace(" km/litre","")
        new_nor['City_Mileage']=new_nor["City_Mileage"].str.replace(",",".")
        
        #creating new column with model and variant 
        new_nor["model specification"] = new_nor['Model'].astype(str) +" "+ new_nor["Variant"].astype(str)

        #options for model 
        brand_options = new_nor['model specification'].unique().tolist()
        sel_model=st.selectbox('Model',brand_options)

        #average petrol and diesel prices
        pet_pr=101.94
        diesel_pr=87.89

        new_nor.drop(new_nor.columns.difference(['Ex-Showroom_Price','model specification','Make','City_Mileage']), 1, inplace=True)
        new_nor.dropna(axis=0,inplace=True)

        new_nor=new_nor[new_nor['model specification']==sel_model] 
        nor_price = new_nor['Ex-Showroom_Price'].values[0]

        #fetching mileage for both cars from dataframe
        nor_mil=new_nor[new_nor['City_Mileage']==sel_model]
        nor_mil = new_nor['City_Mileage'].values[0]

        #input widget for mileage, takes default value from data based on input 
        st.number_input('Mileage',value=float(nor_mil))

        #input widget for km 
        dist = st.number_input('Annual usage in Kilometers')

        cost_km=pet_pr/float(nor_mil)
        tot_cost=dist*cost_km
        submit=st.button('submit') 
    col5, col6 = st.columns([3, 3])    
    with col5:
        if submit:
            st.success('Cost per km: '+ str(cost_km))
            st.success('Total Cost: '+ str(tot_cost))
      

    with col2:
        #input widget for electric car make and model
        sel_evmake=st.selectbox('Manufacturer',ev_options)
        new_ev=ev_com.loc[ev_com['Make'] == sel_evmake]
        brandev_options = new_ev['Model'].unique().tolist()
        sel_evmodel=st.selectbox('Model',brandev_options)
    
        #dropping unecessary columns 
        new_ev.drop(new_ev.columns.difference(['ev_price','Model','Make','Range_Km']), 1, inplace=True)

        #fetching prices for both cars from dataframe    
        ev_price=new_ev[new_ev['Model']==sel_evmodel] 
        ev_price = new_ev['ev_price'].values[0]

        #fetching mileage and range for both cars from dataframe
        ev_range=new_ev[new_ev['Range_Km']==sel_evmodel] 
        ev_range = new_ev['Range_Km'].values[0]

        #input widget for range, takes default value from data based on input 
        rangeev=st.number_input('Range in km',value=float(ev_range))

        #input widget for km , takes default value from data based on input 
        distev = st.number_input('Annual usage in km',value=dist)

        bat_cap=40 #per KWH
        elec_cost=8    # Rs per KWH
        cost_kmev=(bat_cap*elec_cost)/rangeev

        tot_costev=cost_kmev*distev

        st.write("")
        st.write("")
        st.write("")
    

    with col6:
        if submit:

            st.success('Cost per km: '+ str(cost_kmev))
            st.success('Total Cost: '+ str(tot_costev))


    col3, col4 = st.columns([3, 3])
    if submit:
        col3.subheader("Comparision of Initial Prices")
        col4.subheader("graph 2")

    with col3:
        #graph costprice 
  
        if submit:
            def bar_chart():
        #Creating the dataset
                data = {'con':float(nor_price), 'ev':float(ev_price)}
                Courses = list(data.keys())
                values = list(data.values())
                ax=plt.axes()

        # Set color

                ax.set_facecolor('pink')
                fig6 = plt.figure(figsize = (10, 5))

                plt.bar(Courses, values, color="black")
                plt.xlabel("Programming Environment")
                plt.ylabel("Number of Students")
                plt.title("Students enrolled in different courses")
                st.pyplot(fig6)
            bar_chart()

        with col4:
            #graphs for ev vs normal
            l_name_normal=[sel_make+sel_model for i in range(10)]
            l_name_ev=[sel_evmake+sel_evmake for i in range(10)]
            l_row1=l_name_normal+l_name_ev
            l_savings_normal=[tot_cost*i for i in range(1,11)]
            l_savings_ev=[tot_costev*i for i in range(1,11)]
            l_savings=l_savings_normal+l_savings_ev
            l_years=[2022+i for i in range(0,10)]
            l_years=l_years+l_years

            res_df=pd.DataFrame(columns=['model','year','savings'])
            for i in range(20):
                res_df.loc[i]=pd.Series({'model':l_row1[i],'year':l_years[i],'savings':l_savings[i]})
            if submit:

                fig4 = px.bar(res_df, x="model", y="savings", color="model",range_y=[0,200000], animation_frame="year", animation_group="model")
                fig4.update_layout(width=800)
                st.write(fig4)

  
           

        




    




    





