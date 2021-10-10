import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text \
    import CountVectorizer,TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,davies_bouldin_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from scipy import sparse

# fonctions

def preprocessing_text(serie,stop_w_perso): # prend une serie avec du texte et le nettoie
    serie2=serie.str.lower() #minuscules
    tok=RegexpTokenizer('[a-zA-Z]+')
    serie3=[tok.tokenize(description) for description in serie2] #tokenisation
    serie4=[[mot for mot in texte if mot not in stopwords.words('english')]\
         for texte in serie3] #suppression des stop words
    stemmer=SnowballStemmer(language='english')
    serie5=[[stemmer.stem(mot) for mot in texte] for texte in serie4] #stemming
    serie6=[[mot for mot in texte if mot not in stop_w_perso] for texte in serie5] #stemming
    serie7=[' '.join(texte) for texte in serie6] #concatenation de tous les tokens    
    return serie7, serie6 #retourne les textes tokenisés et les textes non tokenisés

def lance_kmeans(serie, clusters, verb,ninit): #lance un kmeans
    km=KMeans(n_clusters=clusters, random_state=0, n_init=ninit, verbose=verb)
    clusters=km.fit_predict(serie)
    centroi=km.cluster_centers_
    return clusters,centroi #retourne les clusters et les centroides

def lance_lda(serie, clusters): # lance un Latent Dirichlet Allocation
    ld=LatentDirichletAllocation(n_components=clusters,random_state=0, max_iter=30, max_doc_update_iter=200)
    docu=ld.fit_transform(serie)
    compo=ld.components_
    nbclusters=pd.DataFrame(docu).apply(lambda x : np.argmax(x) if x.idxmax()!=x.idxmin() else -1, axis=1)
    #en cas d'égalité de toutes les proba categorisation en cluster -1
    return nbclusters, compo, docu #retourne les clusters, les proba mots/sujets et les proba textes/sujets

def transformation2(serie,ngram,idf,mn,mx,b): #transforme un texte en bag of words
    vectorizer= CountVectorizer(token_pattern=r'[a-zA-Z]+',ngram_range=ngram, strip_accents='ascii', min_df=mn, max_df=mx, binary=b)
    tab_vect=vectorizer.fit_transform(serie)
    mots=vectorizer.vocabulary_
    mots2=vectorizer.get_feature_names()
    st_w=vectorizer.stop_words_
    transfo=TfidfTransformer(use_idf=idf)    
    resultats=transfo.fit_transform(tab_vect)
    if idf==True:
        tab_idf=transfo.idf_
    else:
        tab_idf=[]
    return resultats, mots, mots2, tab_idf, tab_vect,st_w #retourne le bag of words, les mots retenus avec index, 
    #les mots retenus sans index, le vecteur idf, le bag of words avant passage par TfidfTransformer et les stop words
 
def lblencoding(categ): #fait un label encoding de variables catégorielles
    lbl=LabelEncoder()
    catego=lbl.fit_transform(categ)
    return catego

def colonnes_categories(serie): #split la colonne product_category_tree en plusieurs colonnes
    #correspondant aux niveaux de l'arborescence
    categories=serie.str.replace('"','').str.replace('[','')\
        .str.replace(']','').str.split('>>', expand=True)[0]
    categories2=serie.str.replace('"','').str.replace('[','')\
        .str.replace(']','').str.split('>>', expand=True)[1]
    categories3=serie.str.replace('"','').str.replace('[','')\
        .str.replace(']','').str.split('>>', expand=True)[2].fillna('UNKNOWN')
    return categories, categories2,categories3 #retourne les 3 premiers niveaux

def supprimer_mots(serie,mots_a_retirer): #remplace des mots d'une liste de mots par des ''
    for mot in mots_a_retirer :
        serie=serie.str.replace(mot,' ')
    return serie


def phase_de_tests(par, serie, categ, clusters,k): #lance une phase de test d'hyperparamètres pour la constitution d'un bag of words
    #utilisé ensuite pour un clustering avec LDA ou kmeans. Les hyperparamètres retenus sont ceux qui aboutissent
    # à l'ARI maximal des clusters obtenus vs catégories
    tab_res_modele=[]
    for i in par['ngram'] :
        for j in par['idf'] :
            for maximum in par['maxdf']:
                for minimum in par['mindf']:
                    for binar in par['bina']:
                        resultats, _,liste_mots2,_,_,_ =transformation2(serie,i,j,minimum,maximum,binar)     
                        if k=='lda':
                            res_clus,compo,docu=lance_lda(resultats, clusters)
                            qualite_cluster = adjusted_rand_score(res_clus,categ)
                            res_lda={'clusters':res_clus, 'idf':j,'ngram':i,
                            'min_df':minimum, 'max_df':maximum, 'binary':binar,
                            'ARI':qualite_cluster}
                            tab_res_modele.append(res_lda)
                            print('ARI pour algo {}, ngram {}, use_idf {}, min_df {}, max_df {}, bin {}:{} max'.\
                            format(k,i,j,minimum, maximum,binar,round(qualite_cluster,2)))
                        elif k=='kme':
                            res_clus,_=lance_kmeans(resultats, clusters,0,3)
                            qualite_cluster = adjusted_rand_score(res_clus,categ)
                            res_kmeans={'clusters':res_clus,'idf':j,'ngram':i,
                            'min_df':minimum, 'max_df':maximum, 'binary':binar, 'ARI':qualite_cluster}
                            tab_res_modele.append(res_kmeans)                       
                            print('ARI pour algo {}, ngram {}, use_idf {}, min_df {}, max_df {}, bin {}:{}'.\
                            format(k,i,j,minimum, maximum,binar,round(qualite_cluster,2)))
                        df_res_modele=pd.DataFrame(tab_res_modele)
                        meill_hp=df_res_modele['ARI'].argmax()
    print(df_res_modele.loc[meill_hp])
    return(df_res_modele, meill_hp) #retourne un dataframe avec l'ensemble des résultats et l'index du test le plus concluant


def lance_tsne(df, perp): # lance un tsne avec un niveau de perplexité donné
    ts_viz=TSNE(perplexity=perp)
    return ts_viz.fit_transform(df)

def heatmap(data, cat1,cat2): # affiche un heatmap à partir de 2 colonnes catégorielles d'un dataframe
    plt.figure()
    sns.heatmap(data[[cat1,cat2]]\
        .pivot_table(columns=cat1,index=cat2,aggfunc=len)\
        .fillna(0),annot=True, yticklabels=True,\
        xticklabels=True,cbar=False, fmt='.0f')
    plt.title('Nombre de documents par cluster / catégorie')
    plt.show(block=False)
    
def top10(df_analyse_mots): #par sujet affiche le top10 des mots qui apparaissent le plus
    relation_mots_sujets=df_analyse_mots.iloc[:,2:9]
    fig, axs=plt.subplots(1, len(relation_mots_sujets.columns), sharex='all')
    plt.suptitle('Top 15 des mots par cluster', y=1)
    for i in range(len(relation_mots_sujets.columns)):
        relation_mots_sujets=relation_mots_sujets.sort_values(i, ascending=False)
        axs[i].barh(np.arange(0,10,1),relation_mots_sujets[i].head(10))
        axs[i].set_yticks(np.arange(0,10,1))
        axs[i].set_yticklabels(relation_mots_sujets.index)
        axs[i].set_xlabel("Nombre d'assignements \nau sujet")
        axs[i].set_title('Cluster {}'.format(i))
        axs[i].invert_yaxis()
    plt.tight_layout()
    plt.show(block=False)

def ajout_colonnes (data): #crée 3 colonnes catégories à partir de la colonne 'product_category_tree'
    # corrige 2 fautes d'ortographe parmi les intitulés, créé 3 colonnes avec des valeurs numériques
    # à partir des 3 colonnes catégories
    data['categories'], data['categories2'],data['categories3']\
        =colonnes_categories(data['product_category_tree'])
    data.loc[data['categories2']==' Living','categories2']=' Living '
    data.loc[data['categories2']==' Showpiece ','categories2']=' Showpieces '
    data['categories_num']=lblencoding(data['categories'])
    data['categories_num2']=lblencoding(data['categories2'])
    data['categories_num3']=lblencoding(data['categories3'])
    data['categories1et2']=data['categories']+'-'+data['categories2']
    data['categories123']=data['categories']+'-'+data['categories2']+'-'+data['categories3']
    return data

def crea_df_analyse_mots (data,bag_of_words,tab_idf,compo,mots_sans_index, nb_cat, ngram,idf,binary,libcolcat):
    #retourne un dataframe pour l'analyse des relations mots/sujets à partir d'hyperparamètres de CountVectorizer, d'un bag of words
    # et des poids sujets/mots retournés par un LDA
    bow=bag_of_words.T.sum(axis=1) #nombre d'occurences par mots
    df_analyse_mots=pd.DataFrame(bow, columns=['mod'])\
                .join(pd.DataFrame(tab_idf,columns=['idf']).applymap(lambda x : round(x,2)))\
                .join(pd.DataFrame(compo.T).applymap(lambda x : round(x,2)))
    df_analyse_mots.index=mots_sans_index
    df_analyse_mots['t_seg']=df_analyse_mots.iloc[:,2:].sum(axis=1)
    for i in nb_cat: #mots par catégorie
        _,_,msi_categ,_,bag_of_w_categ,_= transformation2\
        (data.loc[data[libcolcat]==i,'corpus'],\
                  ngram,idf,1,1.0,binary)
        bow_cat=bag_of_w_categ.T.sum(axis=1) #nombre d'occurences par mots
        df_analyse_mots_cat=pd.DataFrame(bow_cat, columns=[i])
        df_analyse_mots_cat.index=msi_categ
        df_analyse_mots=df_analyse_mots.join(df_analyse_mots_cat, how='left')
    df_analyse_mots=df_analyse_mots.fillna(0)
    df_analyse_mots=df_analyse_mots.rename(columns\
    ={'Home Furnishing ': 'HomeF','Home Decor & Festive Needs ':'HomeD&F',\
        'Beauty and Personal Care ':'Beauty','Kitchen & Dining ':'Kitch',\
            'Computers ':'Comp','Baby Care ':'BabyC','Watches ':'Watch'})
    return df_analyse_mots

def tsne_complet(data,libcolcat, col_clusters,featu): #tests plusieurs valeurs de perplexité pour qu'un TSNE permette
    #une visualisation avec le meilleur coefficient silhouette possible. Crée le TSNE avec cette valeur ensuite
    Perp=[]
    Coef_sil=[]
    tsn=[]
    resultat={}
    colo=[libcolcat,col_clusters]
    for i in range(5,85,10):
        test_tsne=lance_tsne(featu,i)
        indic=silhouette_score(test_tsne,data[col_clusters])
        print('Coefficient de Silhouette pour une perplexité de {} : {}'.format(i,indic))
        Perp.append(i)
        Coef_sil.append(indic)
        tsn.append(test_tsne)
    resultat['Perp']=Perp
    resultat['Silhouette']=indic
    resultat['TSNE']=tsn
    df_test_TSNE=pd.DataFrame(resultat)
    df_test_TSNE=df_test_TSNE.set_index('Perp')
    df_test_TSNE
    good_tsne=df_test_TSNE.loc[df_test_TSNE['Silhouette'].idxmax(),'TSNE']
    df_tsne=pd.DataFrame(good_tsne)
    df_tsne.columns=(['x','y'])
    df_tsne=df_tsne.join(data[libcolcat]).join(data[col_clusters])
    return df_tsne

def affichage_tsne(ts): #affiche TSNE sous format scatterplot avec les catégories en couleur
    sns.scatterplot(data=ts.sort_values('categories'),x='x',y='y',hue='categories', palette='Set1')
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks([])
    plt.yticks([])
    plt.show(block=False)

def categ_tsne(tsne,cate, col,couleurs): #affiche du tsne en format subplots (1 subplot par catégorie)
    fig, axs=plt.subplots(2,4)
    for i in range(2):
        for j in range(4):
            if len(cate)>j+i*4:
                tsne2=tsne.loc[tsne[col]!=cate[j+i*4]]
                tsne3=tsne.loc[tsne[col]==cate[j+i*4]]
                axs[i,j].scatter(x=tsne2.iloc[:,0],y=tsne2.iloc[:,1],color='lightgrey')
                axs[i,j].scatter(x=tsne3.iloc[:,0],y=tsne3.iloc[:,1], color=couleurs[j+i*4])
                axs[i,j].set_title(cate[j+i*4])
                axs[i,j].set_ylabel('')
                axs[i,j].set_xlabel('')
                axs[i,j].set_xticklabels('')
                axs[i,j].set_yticklabels('')
            else :
                axs[i,j].axis('off')
    plt.show(block=False)

def categ_vs_clus(tsne, cate, clus, col_clus): #affichage du TSNE par catégorie par cluster en format subplots 
    fig,axs=plt.subplots(len(cate),len(clus),sharey=True, sharex=True)
    for i in range(len(cate)):
        for j in range(len(clus)):
            df_t2=tsne.copy()
            df_t2['categories']=df_t2['categories'].apply(lambda x : 'Selected' if x == cate[i] else 'NotSelected')
            df_t2[col_clus]=df_t2[col_clus].apply(lambda x : 'Selected' if x == clus[j] else 'NotSelected')        

            sc1=axs[i,j].scatter(x=df_t2.loc[(df_t2[col_clus]=='Selected') ,'x'],\
                y=df_t2.loc[(df_t2[col_clus]=='Selected') ,'y'],color='black',s=4)
            
            sc2=axs[i,j].scatter(x=df_t2.loc[(df_t2['categories']=='Selected') ,'x'],\
                y=df_t2.loc[(df_t2['categories']=='Selected') ,'y'],color='white',s=4)

            sc3=axs[i,j].scatter(x=df_t2.loc[(df_t2[col_clus]=='Selected') & (df_t2['categories']=='Selected'),'x'],\
                y=df_t2.loc[(df_t2[col_clus]=='Selected') & (df_t2['categories']=='Selected'),'y'],color='lime',s=4)
            axs[i,j].set_xticklabels('')
            axs[i,j].set_yticklabels('')        
            axs[i,j].set_facecolor('lightgrey')
            if j==0:
                axs[i,j].set_ylabel(cate[i],\
                    horizontalalignment='right', rotation='horizontal')
            if i==tsne['categories'].nunique()-1:
                axs[i,j].set_xlabel(clus[j])
    fig.legend((sc1,sc2,sc3),('Clusters','Categories','Clusters et\n Categories'),facecolor='lightgrey')

    plt.show(block=False)

def graph_cat_niv2(data): #graph qui montre la relation entre les catégories principales et le premier niveau de sous-catégorie

    figu,axs=plt.subplots(1,3, sharey=True)
    figu.suptitle('Catégories principales et sous-categories de 1er niveau', fontsize=15)
    df_nb_cat=data[['categories','categories1et2','categories_num3']].groupby(['categories','categories1et2']).count()
    df_nb_cat_piv=df_nb_cat.pivot_table(index='categories',columns='categories1et2').fillna(0)
    data['categories'].value_counts()
    sns.countplot(data=data, y='categories',ax=axs[0], orient='h')
    axs[0].set_xlabel('Nombre de documents', fontsize=12)
    axs[0].set_title('Nombre de documents par catégorie principale')
    df_nb_cat_moy=df_nb_cat.pivot_table(index='categories',columns='categories1et2').mean(axis=1)
    df_nb_cat_rs=df_nb_cat.reset_index()[['categories','categories_num3']]
    sns.boxplot(data=df_nb_cat_rs,y='categories',x='categories_num3',ax=axs[2], orient='h', fliersize=0)
    sns.stripplot(data=df_nb_cat_rs, y='categories',x='categories_num3', linewidth=1, ax=axs[2], orient='h')
    axs[2].set_xlabel('Nombre de documents', fontsize=12)
    axs[2].set_title('Distribution du nombre de documents contenus \ndans les sous-catégories de 1er niveau',\
        fontsize=12)
    df_ct_categ=df_nb_cat.reset_index()[['categories']]
    sns.countplot(data=df_ct_categ,y='categories', orient='h', ax=axs[1])
    axs[1].set_xlabel('Nombre de sous-categories', fontsize=12)
    axs[1].set_title('Nombre de sous-catégories de 1er niveau par catégorie', fontsize=12)
    plt.show(block=False)

def countp(data,colcat): #affiche un counplot avec le nombre de documents par cluster
    plt.figure()
    sns.countplot(data=data.sort_values(colcat), x=colcat)
    plt.title('Nombre de documents par cluster')
    plt.show(block=False)

def eval_nb_clusters(data,min_clus, max_clus,tab_feat_cnn,df_transfo): #calcule l'ARI vs catégorie principale et 1ere sous-catégorie 
    #des clusters obtenus via kmeans sur CNN et LDA sur BOW
    tab_res=[]
    for i in range(min_clus, max_clus):
        kcnn,_=lance_kmeans(tab_feat_cnn,i,0,3)
        ldat,_,_=lance_lda(df_transfo,i-1) 
        ari_kcnn_1=adjusted_rand_score(kcnn,data['categories_num'])
        ari_kcnn_2=adjusted_rand_score(kcnn,data['categories_num2'])
        ari_ldat_1=adjusted_rand_score(ldat,data['categories_num'])
        ari_ldat_2=adjusted_rand_score(ldat,data['categories_num2'])
        sil_kme=silhouette_score(df_transfo,kcnn)
        sil_lda=silhouette_score(df_transfo,ldat)
        dict_resu={'Nb Clusters':i,'ARI Clusters texte vs Cat':ari_ldat_1, 'ARI Clusters texte vs Sous-Cat':ari_ldat_2,
        'ARI Clusters image vs Cat':ari_kcnn_1, 'ARI Clusters image vs Sous-Cat':ari_kcnn_2, 'Silhouette Clusters textes':sil_kme,
        'Silhouette Clusters images':sil_lda}
        tab_res.append(dict_resu)

    df_tab_res=pd.DataFrame(tab_res).set_index('Nb Clusters')
    df_tab_res.iloc[:,:-2].plot(xticks=df_tab_res.index,grid=True,marker='D') #graph sans coef silhouette
    plt.title('ARI vs catégories et sous-catégories\nen fonction du nombre de clusters')
    plt.show(block=False)
    return df_tab_res
