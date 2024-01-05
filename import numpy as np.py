import numpy as np
import pickle
import streamlit as st#create account in Streamlit
#map the data

location_mapping = {
    "Kesarapalli":10,
    "Auto Nagar":12,
    "Poranki": 8,
    "Kankipadu": 5,
    "Benz Circle": 0,
    "Gannavaram": 2,
    "Rajarajeswari Peta": 9,
    "Gunadala": 4,
    "Gollapudi": 3,
    "Enikepadu": 1,
    "Vidhyadharpuram": 11,
    "Penamaluru": 7,
    "Payakapuram": 6
}

direction_mapping = {"None": 1,
                     "East": 0,
                     "West":8 ,
                     "NorthEast": 3,
                     "NorthWest":4,
                     "North":2,
                     "South":5,
                     "SouthEast":6,
                     "SouthWest":7}
                     
property_type_mapping = {"Apartment": 0,
                         "Independent Floor": 1,
                         "Independent House": 2,
                         "Residential Plot": 3,
                         "Studio Apartment":4,
                         "Villa" : 5}
                         
                         
#reading pickle file
with open("model.pkl",'rb')as f:
    model=pickle.load(f)
    
    
#create a function to accept inputs and create an array
def predict(bed,bath,loc,size,facing,Type):
    """Function to accept data"""
    selected_location = location_mapping[loc]
    selected_direction = direction_mapping[facing]
    selected_property = property_type_mapping[Type]
    input_data = np.array([[bed,bath,selected_location,
                            size,selected_direction,
                            selected_property]])
    return model.predict(input_data)[0]

if __name__ == "__main__":
    st.header("House Price Prediction")
    col1, col2 = st.columns([2, 1])
    bed = col1.slider("No.of Bedrooms",max_value=10,min_value=1,
                      value=2)
    bath = col1.slider("No.of Bathrooms",max_value=10,min_value=0,
                       value=2)
    loc = col1.selectbox("Select a Location",list(location_mapping.keys()))
    size = col1.number_input("Area",max_value=10000,
                             min_value = 500,value=1000,step=500)
    facing = col1.selectbox("Select a Facing",list(direction_mapping.keys()))
    Type = col1.selectbox("Select Property Type",
                          list(property_type_mapping.keys()))
    result = predict(bed,bath,loc,size,facing,Type)
    submit_button = st.button("Submit")
    if submit_button:
        larger_text = f"<h2 style='color: blue;'>The Predicted House Price is : {format(result, '.2f')} Lakhs</h2>"
        st.markdown(larger_text, unsafe_allow_html=True)