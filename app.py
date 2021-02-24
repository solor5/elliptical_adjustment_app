import streamlit as st
import io
from pathlib import Path
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
from io import BytesIO

##FUNCIONES
def ajuste(df):
  xy=df.iloc[:,0:2].values #puntos X e Y
  Dx=[]
  for i in range(0,xy.shape[0],1):
    x=xy[i,0] 
    y=xy[i,1]
    Dx.append([x**2, x*y, y**2, x, y, 1])
  M=np.matmul(np.transpose(Dx),np.array(Dx)) #Multiplicación entre la transpuesta de Dx y Dx
  Eva, Eve=np.linalg.eig(M) #autovalores y autovectores
  #Encontrar la posición del menor autovalor
  posicion=np.where(Eva == np.amin(Eva)) #de aquí sale una tuple (tupla)
  #Obteniendo el correspondiente autovector
  coef=Eve[:,posicion[0][0]] #coef igual a "a"
  return coef

def grafica(df, coef, i, carac):
  #GRÁFICA
  delta = 2
  xrange = arange(-(abs(df).max()['X']+5), (abs(df).max()['X']+5), delta)
  yrange = arange(-(abs(df).max()['Y']+5), (abs(df).max()['Y']+5), delta)
  X, Y = meshgrid(xrange,yrange)
  fig, ax = plt.subplots(constrained_layout=True) #borre subplots
  cs = ax.contour(X, Y, (coef[0]*X**2 + coef[1]*X*Y + coef[2]*Y**2 + coef[3]*X + coef[4]*Y + coef[5]), [0], colors='blue') #de nada sirve crear el locals()["cs_" + str(i)]
  cs.collections[0].set_label('fitted curve') # put the label of ax.contour
  ax.scatter(df['X'],df['Y'],c='red',marker=".",linewidth=1, label='points')
  ax.set_xlabel('X')
  ax.set_ylabel('Y',rotation=0)
  ax.grid(color='black', linestyle='dotted', linewidth=0.25)
  ax.set_title(str(i+1) +'° orthogonal projection onto XY-Plane') ####SE TIENE QUE MODIFICAR
  ax.legend(fontsize = 8.5)

  textstr = 'Parameters\n$\mathrm{Xc}=%.3f$ mm\n$\mathrm{Yc}=%.3f$ mm\n$\mathrm{Major\ radius}=%.3f$ mm\n$\mathrm{Minor\ radius}=%.3f$ mm\n$\mathrm{Angle}=%.3f$°'%(carac[0], carac[1], carac[2], carac[3], carac[4])  
  # these are matplotlib.patch.Patch properties
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

  # place a text box in upper left in axes coords
  ax.text(0.025, 0.975, textstr, transform=ax.transAxes, fontsize=8.5,
  verticalalignment='top', bbox=props)

  #OBTENCIÓN DE PUNTOS
  v = cs.collections[0].get_paths()[0].vertices #solo coordenadas X e Y (a partir de la gráfica anterior)
  #Hallar la coordenadas Z
  regr = linear_model.LinearRegression()
  regr.fit(np.reshape(df['X'].values, (-1, 1)), np.reshape(df['Z'].values, (-1, 1)))
  z_pred = regr.predict(v[:,0:1])
  v = np.append(v, z_pred, axis=1) #X, Y and Z coordenates
  section = (np.ones(z_pred.shape, dtype=int))*(i+1) #SOLUCIÓN: crear una matriz de unos y multiplicarlos por i+1 [tiene que ser string]
  res = np.array([[str(ele)+'°' for ele in sub] for sub in section])
  df_out=pd.DataFrame(np.append(v, res, axis=1), columns=('X','Y','Z','section')) #dataframe con columnas (o->output)
  return fig, v, df_out

def caracteristicas(coef):
  a=coef[0]; b=coef[1]/2; c=coef[2]; d=coef[3]/2; f=coef[4]/2; g=coef[5]
  Xc=(c*d-b*f)/(b**2-a*c)
  Yc=(a*f-b*d)/(b**2-a*c)
  ac=math.sqrt((2*(a*f**2+c*d**2+g*b**2-2*b*d*f-a*c*g))/((b**2-a*c)*((math.sqrt((a-c)**2+4*b**2))-(a+c))))
  bc=math.sqrt((2*(a*f**2+c*d**2+g*b**2-2*b*d*f-a*c*g))/((b**2-a*c)*(-1*(math.sqrt((a-c)**2+4*b**2))-(a+c))))
  if b==0 and a<c:
    alpha=0
  elif b==0 and a>c:
    alpha=90
  elif b!=0 and a<c:
    alpha=(math.atan(2*b/(a-c))/2)*(180/math.pi) 
  elif b!=0 and a>c:
    alpha=(math.pi/2+(math.atan(2*b/(a-c)))/2)*(180/math.pi) #alpha sale en grados sexagesimales
  carac = [Xc, Yc, ac, bc, alpha]
  return carac

def get_table_download_link(v): 
  v = pd.DataFrame(v).to_csv(index=False)
  b64 = base64.b64encode(v.encode()).decode()  # some strings <-> bytes conversions necessary here
  return f'<a href="data:application/octet-stream;base64,{b64}" download="points.dat">Download DAT file</a>'

def get_image_download_link(fig):
	buffered = BytesIO()
	fig.savefig(buffered, format="png", dpi=300)
	img_str = base64.b64encode(buffered.getvalue()).decode()
	#href = f'<a href="data:file/jpg;base64,{img_str}">Download result</a>'
	return f'<a href="data:file/jpg;base64,{img_str}" download="graph.png">Download graph</a>'

#APLICACIÓN
st.image('https://raw.githubusercontent.com/solor5/femoral_segmentator/main/logo.png', use_column_width=True)
st.title('Femoral Segmentator')
st.write('Program created by William Solórzano, with the support of Ph.D. Carlos Ojeda and Ph.D. Andrés Díaz Lantada.') 
st.write('This program allows the elliptical adjustament from input points and return the coefficients of the XY projection, its parameters and the 3D points of the elliptical section to import into NX')
st.write('General equation of ellipse:')
st.latex('Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0')
#st.latex('z = Gx + H')
st.image('https://raw.githubusercontent.com/solor5/femoral_segmentator/main/ppp3.png', use_column_width=True)

multiple_files = st.file_uploader(
    "Multiple File Uploader",
    accept_multiple_files=True,
    type='dat'
)

count=0
df_united = pd.DataFrame() #dataframe vacio

if len(multiple_files)>0:
  st.header('**Input data**')
  st.write('')
  for file in multiple_files: #cada file en archivos multiples
    file_container = st.beta_expander(
        f"File name: {file.name} (" + str(count+1) + "°)"
    )
    data = io.BytesIO(file.getbuffer())

    #Preprocessing 
    df = pd.read_csv(data, sep=",", names=('X','Y','Z')) #Ingreso con data .dat
    df=df.drop([0,1,2,df.shape[0]-1],axis=0) #las 3 primera columnas no sirven
    df.reset_index(drop=True, inplace=True)
    df = df.astype(float)

    file_container.table(df) #se puede colocar table o write
    locals()["df_" + str(count)] = df.copy() #hasta aquí funciona
    count += 1

  st.write('')
  st.header('**Elliptical adjustment**')
  st.write('')

  for i in range(0,len(multiple_files),1):
    locals()["coef_" + str(i)] = ajuste(locals()["df_" + str(i)]) #ajuste, se obtiene coef
    locals()["carac_" + str(i)] = caracteristicas(locals()["coef_" + str(i)]) #características, se obtiene carac a partir de coef
    locals()["fig_" + str(i)], locals()["v_" + str(i)], locals()["df_" + str(i)] = grafica(locals()["df_" + str(i)], locals()["coef_" + str(i)],i, locals()["carac_" + str(i)]) #cambie esto v -> txt OOOOOO
    st.pyplot(locals()["fig_" + str(i)])
    st.markdown(get_image_download_link(locals()["fig_" + str(i)]), unsafe_allow_html=True)

    df_coef = pd.DataFrame(locals()["coef_" + str(i)], index = ['A', 'B', 'C', 'D', 'E', 'F'], columns=['coefficients'])
    if st.checkbox("Coefficients of " + str(i+1) +'° XY-Plane projection'):
      st.table(df_coef)

    df_prm = pd.DataFrame(locals()["carac_" + str(i)], index = ['Xc', 'Yc', 'Major radius', 'Minor radius', 'Angle'], columns=['parameters'])
    if st.checkbox("Parameters of " + str(i+1) +'° XY-Plane projection'):
      st.table(df_prm)

    df_united = pd.concat((df_united, locals()["df_" + str(i)]), axis=0, ignore_index=True) #funciona
    st.markdown(get_table_download_link(locals()["v_" + str(i)]), unsafe_allow_html=True)

  st.write('')
  st.header('**3D View**')
  st.write('')     
  fig = px.line_3d(df_united, x="X", y="Y", z="Z", color='section')
  st.write(fig)
