import pandas as pd 
import numpy as np 
import torch
import streamlit as st

# Load the data
ratings = pd.read_csv('ratings.csv')
products = pd.read_csv('products.csv')

# Convert the data to tensors
ratings_tensor = torch.Tensor(ratings.values)
products_tensor = torch.Tensor(products.values)


class MLP(torch.nn.Module):
    


    model = MLP()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SDG(model.parameters(), lr=0.01)


    for epoch in range(30):
        predictions = model(ratings_tensor)
        loss = criterion(predictions, ratings_tensor)
        loss.backward()
        optimizer.step()    

# Display the title and subtitle of the app
st.title("Product Rating Prediction App")
st.subheader("Using PyTorch and Streamlit")

# Create a sidebar
sidebar = st.sidebar
sidebar.header("Input Parameters")

# Add some widgets to the sidebar
epochs = sidebar.slider("Number of epochs", 1, 50, 30)
lr = sidebar.slider("Learning rate", 0.001, 0.1, 0.01)
category = sidebar.selectbox("Product category", products['category'].unique())

# Filter the data by the selected category
ratings_filtered = ratings[ratings['product_id'].isin(products[products['category'] == category]['product_id'])]
ratings_tensor_filtered = torch.Tensor(ratings_filtered.values)

# Train the model
st.write(f"Training the model for {epochs} epochs with learning rate {lr}...")
for epoch in range(epochs):
    predictions = model(ratings_tensor_filtered)
    loss = criterion(predictions, ratings_tensor_filtered)
    loss.backward()
    optimizer.step()

# Test the model
with torch.no_grad():
    predictions = model(ratings_tensor[test_indices])
    error = predictions - ratings_tensor[test_indices]
    rmse = torch.sqrt(torch.mean(error**2))
    st.write("RMSE:", rmse)

# Display some recommendations
st.write(f"Here are some recommended products for category {category}:")
recommendations = products[products['category'] == category].sort_values(by='rating', ascending=False).head(10)
st.dataframe(recommendations)
