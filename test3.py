from asyncore import write
from gettext import npgettext
import json

import requests  # pip install requests

from streamlit_lottie import st_lottie


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
st.title("All 'B")
selected = option_menu(
            menu_title=None,  
            options=["Electric vs Conventional Car","Suggest Cars", "Predict Price of Used Car"], 
            default_index=0, 
            orientation="horizontal",
            
        )
            
if selected == "Electric vs Conventional Car":
    #petrol vs electric car analysis
    st.title('Cost Benefit Analysis')


    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    




    col1, col2 = st.columns([3, 3])

    col1.subheader("Select Conventional Car")
    col2.subheader("Select Electric Car")

    #reading normal car and electric car dataset
    #nor_com=pd.read_csv(r"C:\Users\Anusha\Desktop\project\cars_engage_2022.csv")
    nor_com=pd.read_csv("cars_engage_2022.csv")
    #ev_com=pd.read_csv(r"C:\Users\Anusha\Desktop\project\ElectricCar.csv")
    ev_com=pd.read_csv("ElectricCar.csv")

    with col1:
            
        lottie_hello = load_lottieurl("https://assets10.lottiefiles.com/private_files/lf30_skwgamub.json")

        st_lottie(
        lottie_hello,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        
        height=250,
        width=400,
        key=None,
        )
    
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
        new_nor['City_Mileage'].replace('', np.nan, inplace=True)
        new_nor.dropna(subset=['City_Mileage'], inplace=True)
        
        #creating new column with model and variant 
        new_nor["model specification"] = new_nor['Model'].astype(str) +" "+ new_nor["Variant"].astype(str)

        #options for model 
        brand_options = new_nor['model specification'].unique().tolist()
        sel_model=st.selectbox('Model',brand_options)

        sel_fuel=new_nor[new_nor['Fuel_Type']==sel_model]
        sel_fuel= new_nor['Fuel_Type'].values[0]

        


        #average petrol and diesel prices
        if sel_fuel=='Petrol':
            fuel_pr=101.94
        elif sel_fuel=='Diesel':
            fuel_pr=87.89
        else:
            fuel_pr_pr=83

        new_nor.drop(new_nor.columns.difference(['Ex-Showroom_Price','model specification','Make','City_Mileage']), 1, inplace=True)
        new_nor.dropna(axis=0,inplace=True)

        new_nor=new_nor[new_nor['model specification']==sel_model] 
        nor_price = new_nor['Ex-Showroom_Price'].values[0]

        #fetching mileage for both cars from dataframe
        nor_mil=new_nor[new_nor['City_Mileage']==sel_model]
        nor_mil = new_nor['City_Mileage'].values[0]

        #input widget for mileage, takes default value from data based on input 
        sel_mil=st.number_input('Mileage',min_value=float(1), max_value=float(50),value=float(nor_mil))

        #input widget for km 
        dist = st.number_input('Monthly usage in km',min_value=10, max_value=100000,value=1200)

        cost_km=fuel_pr/float(sel_mil)
        tot_cost=dist*cost_km
        submit=st.button('submit') 

    col5, col6 = st.columns([3, 3])    
    with col5:
        if submit:
            cost_km_dis = "{:.2f}".format(cost_km)
            tot_cost_dis = "{:.2f}".format(tot_cost)
            
            st.success('Cost per km: '+ str(cost_km_dis))
            st.success('Total Cost: '+ str(tot_cost_dis))
           
            
      

    with col2:

        lottie_ev = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_tlln0i0q.json")

        st_lottie(
        lottie_ev,
        speed=1,
        reverse=False,
        loop=True,
        quality="low", # medium ; high
        
        height=250,
        width=400,
        key=None,
        )
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
        distev = st.number_input('Monthly usage in kim',min_value=10, max_value=100000,value=dist)

        bat_cap=40 #per KWH
        elec_cost=8    # Rs per KWH
        cost_kmev=(bat_cap*elec_cost)/rangeev

        tot_costev=cost_kmev*distev

        st.write("")
        st.write("")
        st.write("")
    

    with col6:
        if submit:
            cost_kmev_dis = "{:.2f}".format(cost_kmev)
            tot_costev_dis = "{:.2f}".format(tot_costev)
            
            

            st.success('Cost per km: '+ str(cost_kmev_dis))
            st.success('Total Cost: '+ str(tot_costev_dis))
           
    col3, col4 = st.columns([3, 3])
    if submit:
        col3.subheader("Comparision of Ex-showroom Prices")
        col4.subheader("Fuel Expenditure over 10 years")

    with col3:
        
  
        if submit:
            #costprice graph here
            st.write("")
            st.write("")
            st.write("")
           

            data = {'Car Type': [sel_make,sel_evmake], 'Price': [float(nor_price), float(ev_price)]}  
            
            df = pd.DataFrame(data) 
         
            c=alt.Chart(df).mark_bar().encode(x=alt.X('Car Type', sort=None),y='Price')
            
            st.altair_chart(c, use_container_width=True)

    
            with col4:
                #graphs for ev vs normal
                l_name_normal=[sel_make+sel_model for i in range(10)]
                l_name_ev=[sel_evmake+sel_evmodel for i in range(10)]
                l_row1=l_name_normal+l_name_ev
                l_savings_normal=[tot_cost*i for i in range(1,11)]
                l_savings_ev=[tot_costev*i for i in range(1,11)]
                l_savings=l_savings_normal+l_savings_ev
                l_years=[2022+i for i in range(0,10)]
                l_years=l_years+l_years

                res_df=pd.DataFrame(columns=['Model','Year','Fuel Cost'])
                for i in range(20):
                    res_df.loc[i]=pd.Series({'Model':l_row1[i],'Year':l_years[i],'Fuel Cost':l_savings[i]})
                if submit:

                    fig4 = px.bar(res_df, x="Model", y="Fuel Cost", color="Model",range_y=[0,200000], animation_frame="Year", animation_group="Model")
                    fig4.update_layout(width=800)
                    st.write(fig4)            
            


if selected == "Suggest Cars":
    #df = pd.read_csv(r"C:\Users\Anusha\Desktop\project\cars_engage_2022.csv")
    df = pd.read_csv("cars_engage_2022.csv")
    

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

   
    #fuel filter input
    fuel_choice = st.sidebar.selectbox('Select preferred type of Fuel:',Fuel_Type)

    #Mileage filter
    df['City_Mileage'] = df['City_Mileage'].astype(float)
    mileage_choice = st.sidebar.slider('Select preferred mileage range',0,50, (5, 15))
    df_mileage= df['City_Mileage'].between(mileage_choice[0], mileage_choice[1])
    
    #Ex-Showroom_Price filter
    df['Ex-Showroom_Price'] = df['Ex-Showroom_Price'].astype(float)
    price_choice = st.sidebar.slider('Select a price range (in Lakhs)',0,100, (0, 10))
    df_price= df['Ex-Showroom_Price'].between(price_choice[0]*100000, price_choice[1]*100000)
    

    #filtering data based on inputs 
    filter1 = df["Make"].isin([make_choice])
    filter2 = df["Fuel_Type"].isin([fuel_choice])
    filter3 = df_mileage
    filter4 = df_price
    
    
    # displaying data with filters applied
    df=df[filter1 & filter2 & filter3 & filter4]

    comp_df=df.drop(df.columns.difference(['Make','Ex-Showroom_Price','Fuel_Type','City_Mileage','Model','Variant']), 1)
    
    if df.empty:
        st.warning("No cars available with the selected features")
    else:
        st.table(comp_df)
    with st.expander('Click here to view more details'):
        st.write(df)



if selected == "Predict Price of Used Car":
    
    model = pickle.load(open('Usedcar_model2.pkl','rb'))

    #carprice_data = pd.read_csv(r"C:\Users\Anusha\Desktop\project\carprice_data.csv")
    carprice_data = pd.read_csv("carprice_data.csv")


    Make = carprice_data['company'].unique().tolist()
    make_choice=st.selectbox('Select Make:', Make)

    Model = carprice_data['name'].unique().tolist()
    model_choice=st.selectbox('Select Model:', Model)


    year_choice=st.number_input('Select year the car was bought:',min_value=1990,max_value=2021,value=2010)

    Fuel_Type = carprice_data['fuel_type'].unique().tolist()
    fuel_choice=st.selectbox('Select fuel type:',Fuel_Type)

    dist_choice=st.number_input('Select distance travelled:',min_value=1000,max_value=100000,value=40000)


    Model = model
    prediction=Model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                data=np.array([model_choice,make_choice,year_choice,dist_choice,fuel_choice]).reshape(1, 5)))

        

    output = round(prediction[0],2)


    button=st.button('Predict')
    if button:
        display=str( 0.95*output)+" and "+str(1.05*output)
        st.success('Predicted price range is between '+display)




            

            




    




    





