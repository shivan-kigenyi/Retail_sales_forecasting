import streamlit as st
#Building a sales forecasting dashboard
#import libraries to use

import pandas as pd
from prophet import Prophet

import plotly.graph_objects as go

#load the data
df = pd.read_csv('retail_sales_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

#Group sales by product and date
daily_sales = df.groupby(['Date', 'Product Category'])['Total Amount'].sum().reset_index()
product_list = daily_sales['Product Category'].unique()

#Streamlit UI
st.title('Product Sales Forecast Dashboard')

selected_product = st.selectbox('Select a Product Category', product_list)

#prepare data for selected product
product_data = daily_sales[daily_sales['Product Category'] == selected_product]
product_data = product_data.rename(columns = {'Date': 'ds', 'Total Amount':'y'})

#Prophet model
model = Prophet()
model.fit(product_data)

#forecast
future = model.make_future_dataframe(periods =365)
forecast = model.predict(future)

#plot
fig = go.Figure()
fig.add_trace(go.Scatter(x = product_data['ds'], y = product_data['y'], name = 'Actual Sales'))
fig.add_trace(go.Scatter(x = forecast['ds'], y = forecast['yhat'], name = 'Forecast'))

fig.update_layout(title =f'Forecast for {selected_product}', xaxis_title= 'Date',yaxis_title = 'Sales Amount')

st.plotly_chart(fig)
