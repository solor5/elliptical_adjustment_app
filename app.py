import streamlit as st
import numpy as np
from numpy import arange
from numpy import meshgrid
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from PIL import Image
import math
import plotly.express as px
import base64
import io
from io import BytesIO

##FUNCTIONS
def s(n):
  if n>=0:
    sign = '+'
  elif n<0:
    sign = '-'
  return sign

def graph(df, coef, i, prm):
  #GRAPH
  delta = 2
  xrange = arange(-(abs(df).max()['X']+12), (abs(df).max()['X']+12), delta)
  yrange = arange(-(abs(df).max()['Y']+12), (abs(df).max()['Y']+12), delta)
  X, Y = meshgrid(xrange,yrange)
  fig, ax = plt.subplots(constrained_layout=True) 
  cs = ax.contour(X, Y, (coef[0]*X**2 + coef[1]*X*Y + coef[2]*Y**2 + coef[3]*X + coef[4]*Y + coef[5]), [0], colors='blue')
  cs.collections[0].set_label('fitted curve') # put the label of ax.contour
  ax.scatter(df['X'],df['Y'],c='red',marker=".",linewidth=1, label='points')
  ax.set_xlabel('X')
  ax.set_ylabel('Y',rotation=0)
  ax.grid(color='black', linestyle='dotted', linewidth=0.25)
  ax.set_title(str(i+1) +'° orthogonal projection onto XY-Plane')
  ax.legend(fontsize = 8.5)

  #OBTAIN THE COORDINATES
  v = cs.collections[0].get_paths()[0].vertices #X and Y coordinates only from the GRAPH
  #Hallar la coordenadas Z
  regr = linear_model.LinearRegression()
  regr.fit(np.reshape(df['X'].values, (-1, 1)), np.reshape(df['Z'].values, (-1, 1)))
  z_pred = regr.predict(v[:,0:1])
  v = np.append(v, z_pred, axis=1) #X, Y and Z coordinates
  section = (np.ones(z_pred.shape, dtype=int))*(i+1) 
  res = np.array([[str(ele)+'°' for ele in sub] for sub in section])
  df_out=pd.DataFrame(np.append(v, res, axis=1), columns=('X','Y','Z','section')) #dataframe with the column 'section', it allows to obtain the 3D view

  textstr = 'Parameters\n$\mathrm{Xc}=%.3f$ mm\n$\mathrm{Yc}=%.3f$ mm\n$\mathrm{Major\ radius}=%.3f$ mm\n$\mathrm{Minor\ radius}=%.3f$ mm\n$\mathrm{Angle}=%.3f$°'%(prm[0], prm[1], prm[2], prm[3], prm[4])  
  # these are matplotlib.patch.Patch properties
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
  # place a text box in upper left in axes coords
  ax.text(0.025, 0.975, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='top', bbox=props)

  textstr2 = ("Orthogonal projection of the intersection between {:.3f}".format(coef[0]) + '$x^2$ ' + s(coef[1]) + " {:.3f}".format(abs(coef[1])) + '$xy$ ' + s(coef[2]) + " {:.3f}".format(abs(coef[2])) + '$y^2 $' + s(coef[3]) + " {:.3f}".format(abs(coef[3])) + '$x$'
  ' ' + s(coef[4]) + " {:.3f}".format(abs(coef[4])) + '$y$ ' + s(coef[5]) + " {:.3f}".format(abs(coef[5])) + " = 0 \nand $z$ = " + "{:.3f}".format(regr.coef_[0][0]) + "$x$ " + s(regr.intercept_[0]) + " {:.3f}".format(abs(regr.intercept_[0])))
  ax.text(0.5, 0.1, textstr2, transform=ax.transAxes, fontsize=7, verticalalignment='top', horizontalalignment='center', bbox=props)
  return fig, v, df_out


def get_table_download_link(v): 
  v = pd.DataFrame(v).to_csv(index=False)
  b64 = base64.b64encode(v.encode()).decode()  # some strings <-> bytes conversions necessary here
  return f'<a href="data:application/octet-stream;base64,{b64}" download="points.dat">Download DAT file</a>'

def get_image_download_link(fig):
	buffered = BytesIO()
	fig.savefig(buffered, format="png", dpi=300)
	img_str = base64.b64encode(buffered.getvalue()).decode()
	return f'<a href="data:file/jpg;base64,{img_str}" download="graph.png">Download graph</a>'


# AÑADIDOS A PARTIR DEL TFM
## AJUSTE ELÍPTICO 
def elliptical_fit(data):
  Dx=[]
  for i in range(0, data.shape[0], 1):
    x = data[i][0]
    y = data[i][1]
    Dx.append([x**2, x*y, y**2, x, y, 1])
  D = np.dot(np.transpose(Dx),np.array(Dx)) #Multiplication between the transpose of Dx and Dx
  Eva, Eve = np.linalg.eig(D) #eigenvalues(Eva) and eigenvectors(Eve)
  ##Finding the position of the least eigenvalue
  posicion = np.where(Eva == np.amin(Eva)) #tuple
  #Obtaining the corresponding eigenvector
  coef = Eve[:,posicion[0][0]] 
  return coef

def elliptical_parameters(coef):
  a=coef[0]; b=coef[1]/2; c=coef[2]; d=coef[3]/2; e=coef[4]/2; f=coef[5]
  Xc=(c*d-b*e)/(b**2-a*c)
  Yc=(a*e-b*d)/(b**2-a*c)
  R=math.sqrt((2*(a*e**2+c*d**2+f*b**2-2*b*d*e-a*c*f))/((b**2-a*c)*((math.sqrt((a-c)**2+4*b**2))-(a+c))))
  r=math.sqrt((2*(a*e**2+c*d**2+f*b**2-2*b*d*e-a*c*f))/((b**2-a*c)*(-1*(math.sqrt((a-c)**2+4*b**2))-(a+c))))
  if b==0 and a<c:
    alpha=0
  elif b==0 and a>c:
    alpha=90
  elif b!=0 and a<c:
    alpha=(math.atan(2*b/(a-c))/2)*(180/math.pi) 
  elif b!=0 and a>c:
    alpha=(math.pi/2+(math.atan(2*b/(a-c)))/2)*(180/math.pi) #sexagesimal degrees
  prm = [Xc, Yc, R, r, alpha]
  return prm

# Distancia algebraica de la elipse
def elliptical_distance(x, y, coef):
  return ((coef[0]/coef[0])*x**2 + (coef[1]/coef[0])*x*y + (coef[2]/coef[0])*y**2 + (coef[3]/coef[0])*x + (coef[4]/coef[0])*y + (coef[5]/coef[0]))*abs(coef[0])

## RANSAC
# max_iter =  nº máximo de iteraciones
# N = tamaño de la muestra
# data = son las coordenadas (x, y) extraidas de la sección a ajustar
# f_fit = función de ajuste, depende si es un círculo o una elipse
# f_distance =  función de distancia algebraica, depende si es un círculo o una elipse 
def ransac(max_iter, N, data, f_fit, f_distance):

  #####AÑADIDO######
  data=data.iloc[:,0:2].values #X and Y coordinates
  #####AÑADIDO######´


  inliers = np.empty((0, 2)) # inliers, matriz vacia

  l_epoch = []
  l_parm = []
  l_coef = []

  for epoch in range(max_iter):
    ## Muestras
    spr = [] # se almacenan los index que se deben eliminar de la matriz de inliers 
    sample = data[np.random.choice(data.shape[0], N, replace=False), :] # N corresponde al tamaño de la muestra 
    coef_current = f_fit(sample) # ajuste de la muestra
    ## Distancia algebraica con la muestra para determinar que puntos son inliers 
    for i in range(0, sample.shape[0], 1):
      distance = f_distance(sample[i][0], sample[i][1], coef_current)
      if distance <= 0: # para asegurarse que la elipse/círculo queda dentro de la sección interna, propiedades de la distancia algebraica 
        inliers = np.append(inliers, sample[i].reshape((1,2)), axis=0) # reshape para poder agregar datos a la matriz 

    # Eliminar líneas duplicadas en INLIERS      
    inliers = np.unique(inliers, axis=0)
    # Verificación de inliers (segunda comprobación)/ necesito tres puntos para definir el círculo/elipse
    if inliers.shape[0] >= 3: # condición para tener los suficientes puntos y hacer el análisis
      coef_new = f_fit(inliers) # ajuste con los inliers
      # Media y desviación estándar de todos los puntos con el ajuste realizado con los inliers
      mean = np.mean(f_distance(data[:,0], data[:,1], coef_new))
      std = np.std(f_distance(data[:,0], data[:,1], coef_new))
      if mean - std >= 0:
        l_epoch.append(epoch + 1)
        l_parm.append(mean - std)
        l_coef.append(coef_new)
      else:
        for j in range(0, inliers.shape[0], 1):
          distance = f_distance(inliers[j][0], inliers[j][1], coef_new)
          if distance >= 0: #para asegurarse que la elipse queda dentro de la sección, tanto como sea posible
            spr.append(j) # se agregan los index de los puntos que deben eliminarse 
        inliers = np.delete(inliers, spr, axis=0) # para eliminar los puntos 

  # Para encontrar el valor máximo
  max_value = max(l_parm)
  coef_def = l_coef[l_parm.index(max_value)]

  # Para visualizar la gráfica de entrenamiento (SI NO SE QUIERE VISUALIZAR LA GRÁFICA, COMENTAR ESTA PARTE)
  #fig = go.Figure(data=go.Scatter(x=l_epoch, y=l_parm, name = 'Épocas', mode='lines+markers'))
  #fig['layout'].update(height = 600, width = 900, template="plotly_white")
  #fig.show()

  return coef_def

#APP
st.image('https://raw.githubusercontent.com/solor5/elliptical_adjustment_app/main/logo.png', use_column_width=True)
st.title('Elliptical adjustment application')
st.write('Program created by [William Solórzano](https://www.linkedin.com/in/william-solórzano/), with the support of Ph.D. Carlos Ojeda and Ph.D. Andrés Díaz Lantada.')
st.write('This program allows the elliptical adjustment from input points in DAT format. They are obtained from NX by sampling the curve (internal cortical) using points as you can see in step **1**, '
'then export and introduce them into the multiple file uploader. For more details about this sampling process watch this video: https://www.youtube.com/watch?v=EccJgM05Mfc&list=LL&index=17.')
st.image('https://raw.githubusercontent.com/solor5/elliptical_adjustment_app/main/i1.png', use_column_width=True)
st.write('The input file contains X, Y, and Z coordinates of each sample point. X and Y coordinates let the elliptical adjustment of the orthogonal projection of the fitted curve, as consequence the coefficients'
' of the ellipse ($Ax^2 + Bxy + Cy^2 + Dx +Ey +F = 0$) are obtained. The fitted curve is the intersection between elliptical cylinder (adjusted ellipse with Z direction) with a plane ($z = Gx + H$), G and H constants'
' are fitted employing linear regression from the X and Z coordinates (step **2**).')
st.image('https://raw.githubusercontent.com/solor5/felliptical_adjustment_app/main/i2.png', use_column_width=True)
st.write('There are two ways to export the fitted curve to NX. Step **3A** permits that the user obtains the points of the fitted curve in DAT format by clicking on **Download DAT file** then import these points to NX' 
' and with its spline tool get the fitted curve. Likewise, the program provides a 2D graph of the orthogonal projection onto XY-Plane, this graph has its parameters (Xc, Yc, major and minor radius, and the angle),'
' they are introduced to the ellipse tool of NX and finally, the ellipse is projected to the plane to obtain the fitted curve (step **3B**). Furthermore, the program allows downloading the 2D graph of the orthogonal' 
' projection and provides the user with a 3D view of the fitted curves.')
st.image('https://raw.githubusercontent.com/solor5/elliptical_adjustment_app/main/i3.png', use_column_width=True)
st.write('Result (step **4**)')
st.image('https://raw.githubusercontent.com/solor5/elliptical_adjustment_app/main/i4.png', use_column_width=True)
st.write('For more details, please contact us at wsrequejo@gmail.com')
st.write('Download examples of input data: [test1.dat](https://drive.google.com/file/d/1ySmmEaRndP8I50O8w2HTt6dszTK7HnZ1/view?usp=sharing) and [test2.dat](https://drive.google.com/file/d/1E9cR7NHiX1tBGLpGaPQ_c_y5ovv-UfB9/view?usp=sharing)')
multiple_files = st.file_uploader("Multiple File Uploader", accept_multiple_files=True, type='dat')

count=0
df_united = pd.DataFrame() #empty dataframe

if len(multiple_files)>0:
  st.header('**Input data**\n')
  for file in multiple_files: #multiple file uploader
    file_container = st.expander(f"File name: {file.name} (" + str(count+1) + "°)")
    data = io.BytesIO(file.getbuffer())
    #Preprocessing 
    df = pd.read_csv(data, sep=",", names=('X','Y','Z')) #Input data in DAT format
    df=df.drop([0,1,2,df.shape[0]-1],axis=0) #the first three rows are deleted
    df.reset_index(drop=True, inplace=True)
    df = df.astype(float)

    file_container.table(df) 
    locals()["df_" + str(count)] = df.copy() 
    count += 1

  st.header('\n**Elliptical adjustment**\n')

  for i in range(0,len(multiple_files),1):
    #####AÑADIDO######
    n_ite = 300 # número de iteraciones 
    n_sample = 10 # número de muestras   
    locals()["coef_" + str(i)] = ransac(n_ite, n_sample, locals()["df_" + str(i)], elliptical_fit, elliptical_distance)
    st.write(locals()["coef_" + str(i)])
    #####AÑADIDO######
    locals()["prm_" + str(i)] = elliptical_parameters(locals()["coef_" + str(i)]) #parameters, the prm is obtained
    locals()["fig_" + str(i)], locals()["v_" + str(i)], locals()["df_" + str(i)] = graph(locals()["df_" + str(i)], locals()["coef_" + str(i)],i, locals()["prm_" + str(i)]) 
    st.pyplot(locals()["fig_" + str(i)])
    st.markdown(get_image_download_link(locals()["fig_" + str(i)]), unsafe_allow_html=True)

    df_united = pd.concat((df_united, locals()["df_" + str(i)]), axis=0, ignore_index=True) 
    st.markdown(get_table_download_link(locals()["v_" + str(i)]), unsafe_allow_html=True)

    df_prm = pd.DataFrame(locals()["prm_" + str(i)], index = ['Xc', 'Yc', 'Major radius', 'Minor radius', 'Angle'], columns=['parameters'])
    if st.checkbox("Parameters of " + str(i+1) +'° XY-Plane projection'):
      st.table(df_prm)

  st.header('\n**3D View**\n')    
  fig = px.line_3d(df_united, x="X", y="Y", z="Z", color='section')
  st.write(fig)
