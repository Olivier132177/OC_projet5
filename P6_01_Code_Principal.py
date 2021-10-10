from nltk.corpus.reader.wordnet import ADJ
import numpy as np
from numpy.core.shape_base import block
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,silhouette_score
from sklearn.preprocessing import LabelEncoder,Normalizer, OneHotEncoder, StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from tensorflow.python.training.server_lib import ClusterDeviceFilters
import P6_02_Fonctions as fc
from scipy import sparse
import cv2 as cv
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from kmodes.kmodes import KModes
import matplotlib.cm as cm
from PIL import Image

#loading du fichier
path='//home//olivier//Desktop//openclassrooms//P6//'
couleurs=cm.get_cmap('Set1').colors
pathg=path+'Graphs//'
data=pd.read_csv(path+'flipkart_com-ecommerce_sample_1050.csv') 
#création de colonnes catégories (3 premiers niveaux de l'arborescence)
data = fc.ajout_colonnes(data)

nomcolcat='categories_num'
libcolcat='categories'
nb_clusters=data[nomcolcat].nunique()

##############################################################"
# ##########"PARTIE ANALYSE DU TEXTE###########################"

#1ère suppression de mots avant retraitement
mots_a_retirer=['Flipkart.com']
data['description2']=fc.supprimer_mots(data['description'], mots_a_retirer)
fc.graph_cat_niv2(data) #visualisation du nombre de catégories de niveau 2

# liste de mots supplémentaires à supprimer en plus des 
# stopwords nltk, qui a produit les meilleurs résultats
meilleurs_resultats=['rs','product','free','buy','replac','ship','deliveri',\
             'genuin','cash','price','day','guarantee','guarante',\
              'packag','set','onlin','materi','featur','specif','pack',\
             'qualiti','color','type','box','sale','dimens','brand',\
             'best','number','general','model','use','size','made','x',\
              'black','g','length','key','red','yellow','blue','one',\
             'shape','multicolor','also', 'width','cm','print','care','give',\
              'style','water','name','weight','inch','yes','height','boy',\
              'girl','india','men','women','id','bodi','n','gold','like',\
                'green','pink','white','look','multi','light','high','beauti',\
                ]   
stop_w_perso=meilleurs_resultats
#Suppression stopwords/stemming). Retourne 2 colonnes : 1 sous forme de tokens 1 sous forme de texte
data['corpus'],data['corpus_spl']= fc.preprocessing_text(data['description2'], stop_w_perso)
#Retourne 1 colonne avec stemming mais conservation de tous les mots
data['corp_ts_les_mots'],_= fc.preprocessing_text(data['description'], [])

#Phase de test des hyperparamètres
parametres_lda={ 
    'ngram' : [(1,1),(1,2)],
    'idf' : [True,False],
    'maxdf':[1.0],
    'mindf':[1,0.01,0.03,0.035],
    'bina':[True,False]
    }
parametres_kme={ 
    'ngram' : [(1,1),(1,2)],
    'idf' : [True,False],
    'maxdf':[1.0],
    'mindf':[1,0.01,0.035,0.0375],
    'bina':[False,True]
    }
#modelisation LDA
df_res_lda,index_meill_lda=fc.phase_de_tests(parametres_lda, data['corpus'], data[nomcolcat], nb_clusters,'lda')
#modelisation kmeans
df_res_kmeans, index_meill_kme=fc.phase_de_tests(parametres_kme, data['corpus'], data[nomcolcat], nb_clusters,'kme') 
#modelisation et visualisation du LDA avec des paramètres fixes################################
custom=False
if custom == False: #prend les meilleurs hyperparamètres derterminés par le test
    df_transfo, _,mots_sans_index,tab_idf,bag_of_words,st_w =fc.transformation2(data['corpus'],\
        df_res_lda.loc[index_meill_lda,'ngram'],df_res_lda.loc[index_meill_lda,'idf'],\
        df_res_lda.loc[index_meill_lda,'min_df'],df_res_lda.loc[index_meill_lda,'max_df'],\
        df_res_lda.loc[index_meill_lda,'binary'])
elif custom==True: #prend des hyper-paramètres fixés à la main
    df_transfo, _,mots_sans_index,tab_idf,bag_of_words,st_w =fc.transformation2(data['corpus'],\
        (1,1),False,1,1.0,False)
        
clus_id_lda=7 # Selection du nombre de clusters

lda_ideal,compo,docu=fc.lance_lda(df_transfo,clus_id_lda)
data['Clusters_LDA']=lda_ideal
data['Noms Clusters LDA']=data['Clusters_LDA'].apply(lambda x : 'Cluster {}'.format(x))

#######################affichage TSNE#############################
tsne_lda_fait=True
if tsne_lda_fait==False:
    tsne=fc.tsne_complet(data, 'categories','Noms Clusters LDA',df_transfo)
    tsne.to_csv(path+'tsne_lda.csv')
tsne=pd.read_csv(path+'tsne_lda.csv',index_col=0) #loading de la version enregistrée
cate=  np.sort(tsne['categories'].unique())
clus=  np.sort(tsne['Noms Clusters LDA'].unique())

#### affichage des graphs liés au TSNE     ########################

fc.affichage_tsne(tsne)
fc.categ_tsne(tsne,cate, 'categories',couleurs)
fc.categ_tsne(tsne,clus, 'Noms Clusters LDA',couleurs)
fc.categ_vs_clus (tsne, cate, clus, 'Noms Clusters LDA')
fc.heatmap(data,'Noms Clusters LDA','categories')
fc.countp(data,'Clusters_LDA')
fc.heatmap(data,'Noms Clusters LDA','categories1et2')

#modelisation et visualisation du KMEANS avec des paramètres fixes################################

kmeans=False
if kmeans==True :
    df_transfo2, _,mots_sans_index2,tab_idf2,bag_of_words2,_ =fc.transformation2(data['corpus'],\
    df_res_kmeans.loc[index_meill_kme,'ngram'],df_res_kmeans.loc[index_meill_kme,'idf'],\
    df_res_kmeans.loc[index_meill_kme,'min_df'],df_res_kmeans.loc[index_meill_kme,'max_df'],\
    df_res_kmeans.loc[index_meill_kme,'binary'])

    km_ideal,_=fc.lance_kmeans(df_transfo2,7,0,3)
    data['km_ideal']=km_ideal
    data['Noms Clusters Kmeans']=data['km_ideal'].apply(lambda x : 'Cluster {}'.format(x))

    print('ARI :{}'.format(round(adjusted_rand_score(data['km_ideal'],data['categories_num']),3)))
    tsne_kmeans=fc.tsne_complet(data, 'categories','Noms Clusters Kmeans',df_transfo2) #lance un TSNE

    clus2=  np.sort(tsne_kmeans['Noms Clusters Kmeans'].unique())
    cate2=  np.sort(tsne_kmeans['categories'].unique())

    # affichage des graphs liés au TSNE pour la modélisation avec kmeans    ##################################

    fc.affichage_tsne(tsne_kmeans)
    fc.categ_tsne(tsne_kmeans,cate2, 'categories',couleurs)
    fc.categ_tsne(tsne_kmeans,clus2, 'Noms Clusters Kmeans',couleurs)
    fc.categ_vs_clus (tsne_kmeans, cate2, clus2, 'Noms Clusters Kmeans')
    fc.heatmap(data,'Noms Clusters Kmeans','categories')
    fc.countp(data,'Noms Clusters Kmeans')
    fc.heatmap(data,'Noms Clusters Kmeans','categories1et2')

####### Constitution d'un dataframe pour l'analyse des mots

df_analyse_mots=fc.crea_df_analyse_mots(data,bag_of_words,tab_idf,compo,\
    mots_sans_index,data[libcolcat].unique(),df_res_lda.loc[index_meill_lda,'ngram'],\
        df_res_lda.loc[index_meill_lda,'idf'],df_res_lda.loc[index_meill_lda,'binary'],libcolcat)
df_analyse_mots.sort_values('mod', ascending=False).head()
fc.top10(df_analyse_mots) #Top 10 des mots par sujet


############### PARTIE ANALYSE DES IMAGES #######################
#################################################################

#### renaming des images dans un nouveau folder, avec des noms comportant les categories et sous categories####
renaming=False
if renaming==True:
    for i in range(len(data)): 
        img = cv.imread(path+'Images//'+data.loc[i,'image'])
        cv.imwrite(path+'ImagesCat//' + '{}c_{}c_{}.jpg'.format\
            (data.loc[i,'categories_num'],data.loc[i,'categories_num2'],i),img)
    
###### modelisation SIFT ##################"

deja_fait=True
if deja_fait==False:
    bag_of_features=pd.DataFrame()
    for i in range(len(data)): 
        img = cv.imread(path+'Images//'+data.loc[i,'image'])
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create(nfeatures=200, contrastThreshold=0.01)
        kp = sift.detect(gray,None)
        #img2=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv.imwrite('sift_keypoints_{}.jpg'.format(i),img2)
        _,des = sift.compute(gray,kp)
        df_ajout=pd.DataFrame(des)
        df_ajout['image']=i
        bag_of_features=bag_of_features.append(df_ajout)
        print(i)

    bag_of_features.to_csv(path+'BOF.csv') 
   
##################################""
bof=pd.read_csv(path+'BOF.csv') #loading des key points déjà enregistrés
b_images=bof.iloc[:,-1]
bof=bof.iloc[:,1:-1]
bof
deja_fait_norm=True #donne 3 tableaux de descripteurs : 1 brut, 1 normalisé, 1 normalisé avec max à 0,2 et renormalisé

if deja_fait_norm==False:
    norma=Normalizer()
    bof_norm=pd.DataFrame(norma.fit_transform(bof))
    bof_fin=bof_norm.iloc[:,:-1].applymap(lambda x : 0.2 if x >0.2 else x)
    bof_fin=pd.DataFrame(norma.fit_transform(bof_fin))

    tab_bof=[bof,bof_norm,bof_fin] #bof, bof_fin
    tab_km=[]

if deja_fait_norm==False:   #création des clusters pour créer les bags of features (1 par tableau de descripteur)
    for i in range(len(tab_bof)):
        print(' {} a'.format(i))
        resu1,_=fc.lance_kmeans(tab_bof[i],7*200,2,1)
        tab_km.append(resu1)
        
    pd.DataFrame(tab_km).T.to_csv(path+'tab_km.csv')

df_km=pd.read_csv(path+'tab_km.csv', index_col=0) # version déjà enregistrée
df_km_image=pd.concat([b_images,df_km], axis=1)
df_km_image.sort_values('0')

#création des bags of features, 1 par version
bof_pivot=df_km_image[['image','0']].pivot_table(index='image',columns='0', aggfunc=np.count_nonzero).fillna(0)
bof_norm_pivot=df_km_image[['image','1']].pivot_table(index='image',columns='1', aggfunc=np.count_nonzero).fillna(0)
bof_fin_pivot=df_km_image[['image','2']].pivot_table(index='image',columns='2', aggfunc=np.count_nonzero).fillna(0)

choix_clus_sift=7 # choix du nombre de clusters pour le clustering du bag of features

kmeans_bof_pivot,_=fc.lance_kmeans(bof_pivot,choix_clus_sift,0,3)
kmeans_norm_pivot,_=fc.lance_kmeans(bof_norm_pivot,choix_clus_sift,0,3)
kmeans_fin_pivot,_=fc.lance_kmeans(bof_fin_pivot,choix_clus_sift,0,3)

adjusted_rand_score(kmeans_fin_pivot,data['categories_num'])

datag=data.copy()
datag=datag.loc[:,['categories','categories1et2']]
datag['kmeans_fin_pivot']=kmeans_fin_pivot

fc.heatmap(datag,'kmeans_fin_pivot','categories')
tsne_fin_bof=fc.lance_tsne(bof_fin_pivot,20) #visualisation du résultat via TSNE
df_tsne_fin_bof=pd.DataFrame(tsne_fin_bof)
df_tsne_fin_bof.columns=['x','y']
df_tsne_fin_bof=df_tsne_fin_bof.join(data['categories'])
df_tsne_fin_bof

fc.affichage_tsne(df_tsne_fin_bof)

############ CNN #######
##########CNN pour la prédiction uniquement ##############

deja_fait_cnn_pred=True
if deja_fait_cnn_pred==False:
    model = VGG16() # Création du modèle VGG-16 implementé par Keras

    tab_y=[]
    tab_cat=[]
    for i in range(len(data)):
        img = load_img(path+'Images//'+data.loc[i,'image'], target_size=(224, 224))  # Charger l'image
        img2 = img_to_array(img)  # Convertir en tableau numpy
        img2 = img2.reshape((1, img2.shape[0], img2.shape[1], img2.shape[2]))  # Créer la collection d'images (un seul échantillon)
        img2 = preprocess_input(img2)  # Prétraiter l'image comme le veut VGG-16

        y = model.predict(img2)  # Prédire la classe de l'image (parmi les 1000 classes d'ImageNet)
        tab_y.append(y[0])
        tab_cat.append(decode_predictions(y, top=1)[0][0][1])
        print(i)

    model.summary()
    pd.DataFrame(tab_y).to_csv(path+'prob_CNN.csv')
    pd.DataFrame(tab_cat).to_csv(path+'prob_CNN_cat.csv')

    res_tl=pd.read_csv(path+'prob_CNN.csv', index_col=0) #loading de la version déjà créée
    res_tl_cat=pd.read_csv(path+'prob_CNN_cat.csv', index_col=0)

    res_tl['seg']=res_tl.apply(np.argmax, axis=1)
    res_tl['cat']=res_tl_cat #assignation de la catégorie prédite

    res_tl.to_csv(path+'predic_cnn.csv')

res_tl=pd.read_csv(path+'predic_cnn.csv', index_col=0)

adjusted_rand_score(res_tl['seg'],data['categories_num'])
adjusted_rand_score(res_tl['seg'],data['categories_num2'])
adjusted_rand_score(res_tl['seg'],data['categories_num3'])

################CNN pour l'extraction de features############################

deja_fait_ex_feat=True
if deja_fait_ex_feat==False:
    
    model = VGG16()
    # pour retirer le dernier layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    tab_feat_cnn=[]

    for i in range(len(data)):
        image = load_img(path+'Images//'+data.loc[i,'image'], target_size=(224, 224))
        image = img_to_array(image) #transformation en array numpy
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image) #prépare l'image pour le VGG16
        features = model.predict(image) #obtention des features
        tab_feat_cnn.append(features[0])
    pd.DataFrame(tab_feat_cnn).to_csv(path+'ext_feat_cnn.csv')

#######       KMEANS SUR FEATURES CNN ######################"
tab_feat_cnn=pd.read_csv(path+'ext_feat_cnn.csv',index_col=0) #loading de la version déjà enregistrée

meill_clust_cnn = 8 #nombre de clusters sélectionnés
kme_cnn_fin,centro=fc.lance_kmeans(tab_feat_cnn,meill_clust_cnn,0,3)

data['kmeans_CNN']=kme_cnn_fin
data['Noms Clusters CNN']=data['kmeans_CNN'].apply(lambda x : 'Cluster {}'.format(x))

adjusted_rand_score(data['kmeans_CNN'],data['categories_num'])
df_cnn_fin=pd.concat([pd.DataFrame(tab_feat_cnn),pd.Series(kme_cnn_fin)],axis=1)

#######classement des images pour interpretation en fonction des clusters ###################

rangement=False
if rangement==True:
    for i in range(len(data)): 
        img = cv.imread(path+'Images//'+data.loc[i,'image'])
        cv.imwrite(path+'ImagesClus//' + '{}cl_{}_{}c_{}.jpg'.format\
            (data.loc[i,'kmeans_CNN'],data.loc[i,'categories_num'],data.loc[i,'categories_num2'],i),img)
       
####### Visualisation TSNE des résultats ######################################""
tsne_fait=True
if tsne_fait==False:
    tsne_cnn=fc.tsne_complet(data, 'categories','Noms Clusters CNN',df_cnn_fin.iloc[:,:-1])
    tsne_cnn.to_csv(path+'tsne_cnn.csv')

tsne_cnn=pd.read_csv(path+'tsne_cnn.csv', index_col=0) #version enregistrée

cate_cnn=  np.sort(tsne_cnn['categories'].unique())
clus_cnn=  np.sort(tsne_cnn['Noms Clusters CNN'].unique())

###  affichage des graphs liés au TSNE et des heatmaps    #############

fc.affichage_tsne(tsne_cnn)
fc.categ_tsne(tsne_cnn,cate_cnn, 'categories',couleurs)
fc.categ_tsne(tsne_cnn,clus_cnn, 'Noms Clusters CNN',couleurs)
fc.categ_vs_clus (tsne_cnn, cate_cnn, clus_cnn, 'Noms Clusters CNN')
fc.countp(data,'Noms Clusters CNN')
fc.heatmap(data,'Noms Clusters CNN','categories')
fc.heatmap(data,'Noms Clusters CNN','categories1et2')

#########################################################################
#####     CLUSTERING des IMAGES ET des TEXTES combinés      #####################

adjusted_rand_score(data['Clusters_LDA'],data['kmeans_CNN']) #similarité des prédictions LDA texte et kmeans sur CNN
fc.heatmap(data,'kmeans_CNN','Clusters_LDA')

###########  Kmeans sur concatenation des features CNN et du resultat LDA ###########
cnn_lda=tab_feat_cnn.join(pd.DataFrame(docu),rsuffix='Doc')
clsfin,_=fc.lance_kmeans(cnn_lda,8,0,3)
adjusted_rand_score(data['categories_num'],clsfin)

###########    KMODES sur les clusters textes et images ###############################"

choi_clus_kmod=7 #choix du nombre de clusters Kmodes compte tenu des tests

kmod = KModes(n_clusters=choi_clus_kmod, init='Huang', n_init=15, verbose=0,random_state=0)
data['Cluster_Kmodes'] = kmod.fit_predict(data.loc[:,['Clusters_LDA','kmeans_CNN']])
    
#visualisation heatmaps et countplot
fc.heatmap(data,'Cluster_Kmodes','categories')
fc.heatmap(data,'Cluster_Kmodes','categories1et2')
fc.countp(data,'Cluster_Kmodes')
adjusted_rand_score(data['Cluster_Kmodes'],data['categories_num'])

#########################
#test du nombre de clusters 
#idéal pour le kmeans sur CNN et LDA sur BOW vs catégories et sous catégories 1
min_clus=7
max_clus=35
resu_test_clusters=fc.eval_nb_clusters(data,min_clus, max_clus,tab_feat_cnn,df_transfo) 
