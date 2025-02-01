#Importing the required Libraries

import base64
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox,inv_boxcox
from streamlit_option_menu import option_menu

# Setting Page configuration and Background

st.set_page_config(page_title="Copper Modeling", page_icon="Icon Logo.png",layout='wide')


# Set custom CSS for background image
background_image_url = "https://i.pinimg.com/1200x/9e/7e/8d/9e7e8d917b07ca8002fc2d2114598f2b.jpg"  

# Custom CSS to set background image for the entire page
st.markdown(f"""
    <style>
        /* Background and page styling */
        body {{
            background-image: url("{background_image_url}");
            background-size: cover;  /* Ensures the image covers the entire page */
            background-repeat: no-repeat;  /* Ensures the image does not repeat */
            background-attachment: fixed;  /* Makes the background fixed when scrolling */
            color: white;  /* Text color set to white for contrast */
        }}
        
        .stApp {{
            background-color: transparent;  /* Transparent background for Streamlit app */
        }}

        /* Styling for Option Menu Container (works for most Streamlit setups) */
        .css-1n3lx18 {{
            background: rgba(0, 0, 0, 0.4);  /* Semi-transparent background */
            backdrop-filter: blur(1px);  /* Apply blur effect */
            border-radius: 10px;  /* Round the corners */
            padding: 15px 30px;  /* Add padding to the menu */
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);  /* Optional: add shadow for depth */
        }}

        /* Styling for Option Menu Links */
        .css-1n3lx18 .nav-link-selected {{
            background-color: #E5C19F;  /* Highlighted option with Copper White */
            color: black;  /* White text when selected */
        }}

        .css-1n3lx18 .nav-link {{
            color: white;  /* White for unselected options */
        }}

        /* Styling for content text with text-shadow and increased font size */
        .content-text {{
            color: white;
            font-size: 18px;  /* Increased font size */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);  /* Light text shadow effect */
        }}
        
        /* Styling for columns with blur effect */
        .stColumn {{
            background: rgba(0, 0, 0, 0.2);  /* Semi-transparent black background */
            backdrop-filter: blur(10px);  /* Apply blur effect */
            border-radius: 10px;  /* Round the corners */
            padding: 10px;
        }}
    </style>
""", unsafe_allow_html=True)


# Header styling

st.markdown("<h1 style='text-align:center;text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); color:#bf278c; font-size:50px;'>Industrial Copper Modeling</h1>", unsafe_allow_html=True)
    
selected = option_menu(
    menu_title=None,
    options=["Status Prediction","Selling Price Prediction"],
    icons=["house", "bar-chart"],
    default_index=0,
    orientation="horizontal",
    styles={
        "nav-link-selected": {
            "background-color": "#bf278c",  # Highlighted option with Copper White
            "color": "black",  # White text when selected
        },
        "nav-link": {
            "color": "#ffffff",  # white for unselected options
        }
    }
)
# Instantiation

le = LabelEncoder()

# Transforming and Inverse Transforming the Date column for streramlit application

df1 = pd.read_csv('Copper.csv')
df1['item_date'] = le.fit_transform(df1['item_date'])
encoded_item_date = list(df1['item_date'].unique())
original_item_date = list(le.inverse_transform(df1['item_date'].unique()))

def item_date_dic():
    item_date = {}
    for key in original_item_date:
        for value in encoded_item_date:
            item_date[key] = value
            encoded_item_date.remove(value)
            break
    return item_date




#  STATUS PREDICTION
if selected == 'Status Prediction' :
        
        c1,c2 = st.columns(2)
        with c1:
            date = st.selectbox('Transaction Date (yyyy-mm-dd)',options= list(original_item_date))
            user_quant = st.number_input("Item Quantity in Tons(0.00 - 99999.99)")
            code = st.selectbox('Country code',options= list(df1['country'].unique()))
            use_type = st.selectbox('Item Type',options=('W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'))
            application = st.selectbox('Application',options=list(df1['application'].unique()))
        with c2:
            thick = st.number_input('Thickness(0.20 to 2500.00)')
            width = st.number_input('Width(1.0 to 3000.00)')
            prod_ref = st.selectbox('Product Reference Code:',options= list(df1['product_ref'].unique()))
            del_date = st.selectbox('Delivery Date (yyyy-mm-dd)',options= list(df1['delivery date'].unique()))
            sel_pr = st.number_input("Selling Price(0.10 to 82000.00)")
         
        button = st.button('Predict Status')
        

        if button:
            item = item_date_dic()
            item_date = item[date]

            quantity_tons = boxcox(user_quant,0.005921481390060094).round(2)

            item_type = {'W':5,'S':3,'PL':2,'WI':6,'others':1,'IPL':0,'SLAWR':4}
            item_typ = item_type[use_type]

            thickness = boxcox(thick,-0.1792730255842548).round(2)

            delivery_date = int(del_date[0:4])

            selling_price = boxcox(sel_pr,0.09343054475928997).round(2)

            ip = np.array([[item_date, quantity_tons, code, item_typ, application, thickness, width, prod_ref, delivery_date, selling_price]])
            
            with open(r'D:\Industrial Copper Modeling\rf_model.pkl','rb') as file:
                rf_model = pickle.load(file)

            status_predict = rf_model.predict(np.array(ip))

            if status_predict:
                st.markdown('<h2 style="text-align: center; color:#bf278c; text-shadow: 2px 2px 4px #000000;">Transaction or Item Status : WON !</h2>', 
                            unsafe_allow_html=True)
            else:
                st.markdown('<h2 style="text-align: center; color:#bf278c; text-shadow: 2px 2px 4px #000000;">Transaction or Item Status: LOST</h2>', 
                            unsafe_allow_html=True)


# SELLING PRICE PREDICTION
                
if selected == 'Selling Price Prediction'  :
        a1,a2 = st.columns(2)

        with a1:
            date = st.selectbox('Transaction Date (yyyy-mm-dd)',options= list(original_item_date))
            user_quant = st.number_input("Item Quantity in Tons(0.00 - 99999.99)")
            code = st.selectbox('Country code',options= list(df1['country'].unique()))
            use_type = st.selectbox('Item Type',options=('W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR'))
            application = st.selectbox('Application',options=list(df1['application'].unique()))
        with a2:
            thick = st.number_input('Thickness(0.20 to 2500.00)')
            width = st.number_input('Width(1.0 to 3000.00)')
            prod_ref = st.selectbox('Product Reference Code:',options= list(df1['product_ref'].unique()))
            del_date = st.selectbox('Delivery Date (yyyy-mm-dd)',options= list(df1['delivery date'].unique()))
            status = st.selectbox('Status',options= ('Won','Lost'))
        
        button1 = st.button('Predict the Selling Price')
             
        if button1:
            item = item_date_dic()
            item_date = item[date]

            quantity_tons = boxcox(user_quant,0.005921481390060094).round(2)

            item_type = {'W':5,'S':3,'PL':2,'WI':6,'others':1,'IPL':0,'SLAWR':4}
            item_typ = item_type[use_type]

            thickness = boxcox(thick,-0.1792730255842548).round(2)

            delivery_date = int(del_date[0:4])
            
            def stat(status):
                if status == 'Won':
                    return 1
                else:
                    return 0
                

            ip1 = [[item_date,quantity_tons,code,item_typ,application,thickness,width,prod_ref,delivery_date,stat(status)]]

            with open(r'D:\Industrial Copper Modeling\rf_reg.pkl','rb') as file:
                reg_model = pickle.load(file)
            
            price_predict = reg_model.predict(np.array(ip1))
            s_price = inv_boxcox(price_predict[0].round(2), 0.09343054475928997)

            st.markdown(f'<h3 style="text-align: center; color:#bf278c; text-shadow: 2px 2px 4px #000000;">Predicted Selling Price: {s_price}</h3>',
                        unsafe_allow_html=True)
            
