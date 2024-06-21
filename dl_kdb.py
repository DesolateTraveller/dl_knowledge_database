#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------
import plotly.graph_objects as go
from scipy.special import erfc


#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
#import custom_style()
st.set_page_config(page_title="Deep Learning | Knowledge Database",
                   layout="wide",
                   #page_icon=               
                   initial_sidebar_state="auto")
#----------------------------------------
st.title(f""":rainbow[Deep Learning | Knowledge Database | v0.1]""")
st.markdown('Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a>', 
            unsafe_allow_html=True)
st.info('**Disclaimer : :blue[Thank you for visiting the app] | Unauthorized uses or copying of the app is strictly prohibited | Click the :blue[sidebar] to follow the instructions to start the applications.**', icon="ℹ️")
#----------------------------------------
# Set the background image
st.divider()

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)
#-------------------------------
def logistic(z):
    return 1 / (1 + np.exp(-z))
#-------------------------------
def relu(z):
    return np.maximum(0, z)
#-------------------------------
def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)
#-------------------------------
selu_alpha = -np.sqrt(2 / np.pi) / (erfc(1/np.sqrt(2)) * np.exp(1/2) - 1)
selu_scale = (1 - erfc(1 / np.sqrt(2)) * np.sqrt(np.e)) * np.sqrt(2 * np.pi) * (2 * erfc(np.sqrt(2))*np.e**2 + np.pi*erfc(1/np.sqrt(2))**2*np.e - 2*(2+np.pi)*erfc(1/np.sqrt(2))*np.sqrt(np.e)+np.pi+2)**(-1/2)
#-------------------------------
def selu(z, scale=selu_scale, alpha=selu_alpha):
    return scale * elu(z, alpha)
#-------------------------------
def plot_function(func, title, alpha=None):
    fig = go.Figure()
    if alpha:
        fig.add_trace(go.Scatter(x=z, y=func(z, alpha=alpha), mode='lines', line=dict(color='red', width=3)))
    else:
        fig.add_trace(go.Scatter(x=z, y=func(z), mode='lines', line=dict(color='red', width=3)))
    fig.update_layout(title = title, xaxis_title='Z',width=700, height=400,
                            font=dict(family="Courier New, monospace",size=16,color="White"), margin=dict(t=30, b=0, l=0, r=0))
    fig.update_xaxes(zeroline=True, zerolinewidth=3, zerolinecolor='violet')
    fig.update_yaxes(zeroline=True, zerolinewidth=3, zerolinecolor='violet')

    return fig
#-------------------------------
def plot_function_derivative(func, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=derivative(func, z), mode='lines', line=dict(color='red', width=3)))
    fig.update_layout(title = title, xaxis_title='Z', width=700, height=400,
                            font=dict(family="Courier New, monospace",size=16,color="White"), margin=dict(t=30, b=0, l=0, r=0))
    fig.update_xaxes(zeroline=True, zerolinewidth=3, zerolinecolor='violet')
    fig.update_yaxes(zeroline=True, zerolinewidth=3, zerolinecolor='violet')
    return fig

#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

z = np.linspace(-8,8,200)

#-------------------------------------------------------------------

st.sidebar.header("Contents", divider='blue')
#st.sidebar.info('Please choose from the following options and follow the instructions to start the application.', icon="ℹ️")
dl_type = st.sidebar.radio("**:blue[Choose the options]**", ["Activation Functions", 
                                                            "Option2",
                                                            "Option3",
                                                            "Option4",
                                                            "Option5", 
                                                            "Option6",])
st.sidebar.divider()

#---------------------------------------------------------------------------------------------------------------------------------
### Activation Functions
#---------------------------------------------------------------------------------------------------------------------------------

if dl_type == "Activation Functions" :

    #st.subheader("Activation Functions",divider='blue')
    a_f = st.selectbox('**:blue[Choose an activation function]**', ['None', 'Logistic (Sigmoid) Function', 'Hyperbolic Tangent (Tanh) Function', 'ReLU Function', 'LeakyReLU Function', 'Variants of LeakyReLU Function', 'Exponential Linear Unit Function', 'SELU Function'])
    st.divider()

#-----------------------------------------------------------------------
    
    if a_f == 'Logistic (Sigmoid) Function':

        st.subheader("Logistic (Sigmoid) Function",divider='blue')

        #st.subheader('Description')
        st.write('It is a sigmoid function with a characteristic "S"-shaped curve.')
        st.markdown(r'**$sigmoid(z)=\frac{1}{1+exp(-z)}$**')
        st.write('The output of the logistic (sigmoid) function is always between 0 and 1.')   

        st.divider()
    
        #st.subheader('Plot')
        logistic_fig  = plot_function(logistic, title='Logistic (Sigmoid) Activation Function')
        logistic_fig.add_annotation(x=7, y=1, text='<b>Saturation</b>', showarrow=True,font=dict(family="Montserrat", size=16, color="#1F8123"),
                                        align="center",arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A835E1", ax=-20, ay=30,)
        logistic_fig.add_annotation(x=-7, y=0, text='<b>Saturation</b>', showarrow=True,font=dict(family="Montserrat", size=16, color="#1F8123"),
                                        align="center",arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A835E1", ax=0, ay=-30,)
        st.plotly_chart(logistic_fig)
            
        with st.expander('Plot Explanation'):
            st.write('- The logistic function saturates as the inputs become larger (either positive or negative).')
            st.write('- For large positive and negative values, the function gets asymptotically close to 1 and 0, respectively.')
            st.write('- When the function saturates, its gradient becomes very close to zero, which slows down learning.')

        st.divider()
    
        #st.subheader('Derivative')
        st.markdown(r'$sigmoid^{\prime}(z)=sigmoid(z)(1−sigmoid(z))$')
        st.text("")
        logistic_der_fig = plot_function_derivative(logistic, title='Derivative of the Logistic Function')
        st.plotly_chart(logistic_der_fig)

        with st.expander('Plot Explanation'):
            st.write('Notice that the derivative of the logistic function gets very close to zero for large positive and negative inputs.')

        st.divider()

        st.subheader('Pros')
        st.write('1. The logistic function introduces non-linearity into the network which allows it to solve more complex problems than linear activation functions.\n2. It is continuous and differentiable everywhere.\n3. Because its output is between 0 and 1, it is very common to use in the output layer in binary classification problems.')

        st.subheader('Cons')
        st.write("1. Limited Sensitivity\n- The logistic function saturates across most of its domain.\n- It is only sensitive to inputs around its midpoint 0.5.")
        st.write("2. Vanishing Gradients in Deep Neural Networks\n- Because the logistic function can get easily saturated with large inputs, its gradient gets very close to zero. This causes the gradients to get smaller and smaller as backpropagation progresses down to the lower layers of the network.\n- Eventually, the lower layers' weights receive very small updates and never converge to their optimal values.")

#-----------------------------------------------------------------------
    
    if a_f == 'Hyperbolic Tangent (Tanh) Function':

        st.subheader("Hyperbolic Tangent (Tanh) Function",divider='blue')

        st.write('The tanh function is also a sigmoid "S"-shaped function.')
        st.markdown(r'$tanh(z)=\frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$')
        st.write('The range of the tanh function is between -1 and 1.')

        st.divider()
        
        #st.subheader('Plot')
        tanh_fig = plot_function(np.tanh, title='Hyperbolic Tangent Function')
        tanh_fig.add_annotation(x=7, y=1, text='<b>Saturation</b>', showarrow=True,font=dict(family="Montserrat", size=16, color="#1F8123"),
                                align="center",arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A835E1", ax=-20, ay=30,)
        tanh_fig.add_annotation(x=-7, y=-1, text='<b>Saturation</b>', showarrow=True,font=dict(family="Montserrat", size=16, color="#1F8123"),
                                align="center",arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#A835E1", ax=0, ay=-30,)
        st.plotly_chart(tanh_fig)
        
        with st.expander('Plot Explanation'):
            st.write('- The tanh function saturates as the inputs become larger (either positive or negative).')
            st.write('- For large positive and negative values, the function gets asymptotically close to 1 and -1, respectively.')
            st.write('- When the function saturates, its gradient becomes very close to zero, which slows down learning.')
    
        st.divider()

        #st.subheader('Derivative')
        st.markdown(r'$tanh^{\prime}(z)= 1 - (tanh(z))^{2}$')
        st.text("")
        tanh_der_fig = plot_function_derivative(np.tanh, title='Derivative of the Tanh Function')
        st.plotly_chart(tanh_der_fig)
        with st.expander('Plot Explanation'):
            st.write('Notice that the derivative of the tanh function gets very close to zero for large positive and negative inputs.')
    
        st.divider()
        
        st.subheader('Pros')
        st.write("1. The tanh function introduces non-linearity into the network which allows it to solve more complex problems than linear activation functions.\n2. It is continuous, differentiable, and have non-zero derivatives everywhere.\n3. Because its output value ranges from -1 to 1, that makes each layer's output more or less centered around 0 at the beginning of training, whcih speed up convergence.")

        st.subheader('Cons')
        st.write("1. Limited Sensitivity\nThe tanh function saturates across most of its domain. It is only sensitive to inputs around its midpoint 0.")
        st.write("2. Vanishing Gradients in Deep Neural Networks\n- Because the tanh function can get easily saturated with large inputs, its gradient gets very close to zero.\n- This causes the gradients to get smaller and smaller as backpropagation progresses down to the lower layers of the network.\n- Eventually, the lower layers' weights receive very small updates and never converge to their optimal values.")

        st.markdown("**Note**: the vanishing gradient problem is less severe with the tanh function because it has a mean of 0 (instead of 0.5 like the logistic function).")

#-----------------------------------------------------------------------
