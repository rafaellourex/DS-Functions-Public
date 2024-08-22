import pandas as pd
import numpy as np
from scipy import sparse
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity


""" 

ALS Recommender

"""

def ALS_Recommender (model,alpha, data_train,data_val):
    from implicit.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import bm25_weight

    # weight the matrix, both to reduce impact of users that have played the same artist thousands of times
    # and to reduce the weight given to popular items
    data_train = bm25_weight(data_train, K1=100, B=0.8)
    
    from implicit.evaluation import precision_at_k, train_test_split,AUC_at_k,mean_average_precision_at_k
    model_ALS = model
    alpha_val = alpha
    data_conf = (data_train * alpha_val).astype('double')
    # Calculate the confidence by multiplying it by our alpha value.
    # Fit the model
    model_ALS.fit(data_conf,show_progress=False)
    precision = precision_at_k(model_ALS, data_train, data_val, K=10, num_threads=4,show_progress=False)
    AUC = AUC_at_k(model_ALS, data_train, data_val, K=10, num_threads=4,show_progress=False)
    #mean average precision
    MAP = mean_average_precision_at_k(model_ALS, data_train, data_val, K=10, num_threads=4,show_progress=False)
    print(f'Precision at 10 Score : {precision}')
    print(f'MAP at 10 Score : {MAP}')
    print(f'AUC at 10 Score : {AUC}')
    return(model_ALS)



def recommend_ALS (model,user_id,data_matrix, data_train,index_name, N=10 ):
    
    ''''''''''
    The parameters of data_train change depending on the context of the dataset (eg: in the Business Case the 
    product ID is StockCode and the Description is Description)
    
    ''''
    
    data_clean = data_train.copy()
    us_id = np.argwhere(data_matrix.index == user_id)[0][0]
    to_recommend_index , prod_scores = model.recommend(userid=us_id,user_items=x_train_matrix_csr_bought, N=N+30, filter_already_liked_items=False )


    prod_indexes = []
    for i in to_recommend_index:
        prod_indexes.append(data_matrix.columns[i])

    description = []
    for i in prod_indexes:    
        description.append(data_clean.loc[data_clean['StockCode']==i,'Description'].unique())

    bought = data_clean.loc[data_clean[f'{index_name}']==user_id,'StockCode'].unique()
    bought_description = data_clean.loc[data_clean[f'{index_name}']==user_id,'Description'].unique()
    
    rec_index = []
    rec_description = []
    for i,ii in zip(prod_indexes,description):
        if len(rec_index)<10:
            if i not in bought:
                rec_index.append(i)
                rec_description.append(ii[0])
            
            


    print(f'The Customer {user_id} has bought:')
    for index, i in enumerate(bought_description):
        print(f'{index+1} - {i}')
    counter =0

    print('-'*125)

    print(f'The top {10} recommendations are:')
    for i,ii in zip(rec_description,rec_index):
        counter = counter +1
        print(f'{counter} - {i}, with category {ii[:2]}')
    als_rec = pd.DataFrame()
    als_rec['ItemID'] = rec_index
    als_rec['Description'] = rec_description
    return(als_rec.set_index('ItemID'))




""" 

LightFM Recommender

"""


def create_id (data):
    
    """ 
    This functions receives an interaction matrix n * m and create a dictionary with 
    IDs for the customers and items to further be used in LightFm
    
    """

    user = data.index     
    product = data.columns
    
    user_ = {}
    for i,ii in enumerate(user):
        user_[i] = ii
        
    prod_2 = {}
    for i , ii in zip(product_sign.index, product_sign['Description']):
        prod_2[i] = ii
    return(user_,prod_2)

def oneHotEncoder(data):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
    non_metric_features =list(data.select_dtypes(exclude=np.number).set_index(data.index).columns)
    ohc = OneHotEncoder(sparse=False)
    ohc_feat = ohc.fit_transform(data[non_metric_features])
    names = ohc.get_feature_names()
    
    ohc_cat = pd.DataFrame(data =ohc_feat ,columns = names, index = data.index)
    return(ohc_cat)

def create_feature_matrix (data,features):
    """  
    This function receives: 
    The original trnsaction data
    The columns that will be used for the features 
    """
    user_feat = data.copy()
    user_feat = user_feat[features]
    user_feature = csr_matrix(oneHotEncoder_(user_feat))
    return(user_feature)


def lighfm_recommender (model,data_train_matrix, data_val_matrix,data_train,epochs,user_feat=None, k=10):
    
    '''''''''
    The function receives:
    LightFM model 
    Train Matrix (CSR format)
    Validation Matrix (CSR format)
    train_Data (not in CSR format)
    Number of Epochs
    User_Features (optional, its a OneHotEncoded matrix)
    K to calculate Precision and Recall (10 is predefined)
    
    '''''
    
    #this function receives 2 matirxes in csr format and a DF to create the item similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    import lightfm
    from scipy import sparse
    from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
    #ceate and train model  
    model_light=model
    model_light.fit(data_train_matrix,num_threads=4,epochs=epochs,user_features = user_feat)
    
    #create similarity matrix between items
    item_df = sparse.csr_matrix(model_light.item_embeddings)
    item_distance = pd.DataFrame(cosine_similarity(item_df), columns =data_train.columns, index = data_train.columns )

    # Model Assess
    precision_test = precision_at_k(model=model_light,test_interactions=data_val_matrix,user_features=user_feat,k=k,).mean()
    precision_train = precision_at_k(model=model_light,test_interactions=data_train_matrix,user_features=user_feat,k=k,).mean()
    recall_test = recall_at_k(model=model_light,test_interactions=data_val_matrix,user_features=user_feat,k=k).mean()
    recall_train = recall_at_k(model=model_light,test_interactions=data_train_matrix,user_features=user_feat,k=k,).mean()
    auc_train = auc_score(model=model_light, test_interactions=data_train_matrix,user_features=user_feat).mean()
    auc_test = auc_score(model=model_light,test_interactions=data_val_matrix,user_features=user_feat).mean()

    print(f"Train AUC Score: {auc_train}")
    print(f"Test AUC Score: {auc_test}")
    print('-'*125)
    print(f'Train Precision at {k} Score: {precision_train}')
    print(f'Test Precision {k} Score: {precision_test}')
    print('')
    print(f'Train Recall {k} Score: {recall_train}')
    print(f'Test Recall {k} Score: {recall_test}')
    return(model_light,item_distance)


#https://towardsdatascience.com/recommendation-system-in-python-lightfm-61c85010ce17
def sample_recommendation_user(model, interactions, interaction_id, user_dict, 
                               item_dict,threshold = 0,nrec_items = 5, show = True,user_feat = None):
    
    n_users, n_items = interactions.shape
    interaction_id = interaction_id
    user_id = user_idx[interaction_id]
    scores = pd.Series(model.predict(interaction_id,np.arange(n_items),user_features=user_feat))
    scores.index = interactions.columns
    
    
    known_items = list(pd.Series(interactions.loc[user_id,:] \
                                 [interactions.loc[user_id,:] > threshold].index).sort_values(ascending=False))
    scores = scores.sort_values(ascending=False)
    to_recommend = []
    score = []
    description = []
    
    ''''''''''
    The parameters of data_train change depending on the context of the dataset (eg: in the Business Case the 
    product ID is StockCode and the Description is Description)
    
    ''''
    
    appended = []
    for i,ii in zip(scores.index, scores):
        if len(appended)<15:
            if i not in known_items:
                to_recommend.append(i)
                score.append(ii)
                description.append(data_clean.loc[data_clean['StockCode'].isin([i]),'Description'].unique()[0])
                appended.append(0)

        
    
    lightRecom = pd.DataFrame()
    lightRecom['ProductID'] = to_recommend
    lightRecom['Description'] = description
    lightRecom['Score'] = score
    display(lightRecom.iloc[:10])
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
        print ("User: " + str(user_id))
        print("Has Previosly Bought:")
        counter = 1
        for i in known_items:
            print(str(counter) + '- ' + i)
            counter+=1
            
            
        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
    return(lightRecom)


def recommend_with_items (similarities, item_Id,k):
    #this function receives a similarity matrix and recommends k items based on a target item_id
    
    
    '''''''''
    The function receives:
    Similarity Matrix (item to item)
    Item ID 
    Number of items to recommend
    
    ''''
    
    ''''''''''
    The parameters of data_train change depending on the context of the dataset (eg: in the Business Case the 
    product ID is StockCode and the Description is Description)
    
    ''''
    
    
    target_item = data_clean.loc[data_clean['StockCode']==f'{item_Id}','Description'].unique()
    scores = similarities[f'{item_Id}'].sort_values(ascending=False)[1:k+1]
    scores_id = scores.index
    description = []
    for i in scores_id:
        id_ = data_clean.loc[data_clean['StockCode']==f'{i}','Description'].unique()
        description.append(id_[0])
    to_recommend = pd.DataFrame(index = scores_id)
    to_recommend['Description'] = description
    to_recommend['Score'] = scores.values
    print(f'The {k} most similar item to item {target_item[0]} are:')
    for i,ii in enumerate(description):
        print(f'{i+1} - {ii}')
    return(to_recommend)
    

def cold_start_rec (model,user_id,item_dict, user_feature,nrec_items,columns):
    
    '''''''''
    The function receives:
    Model 
    UserID we want to predict
    Item Dict
    User Feature (this time mandatory)
    Nr items to recommend 
    Columns regarding the items 
    ''''
    
    ''''''''''
    The parameters of data_train change depending on the context of the dataset (eg: in the Business Case the 
    product ID is StockCode and the Description is Description)
    
    ''''
    
    
    scores= model.predict(user_id,item_ids=np.arange(len(columns)), user_features=user_feature)
    products = columns
    scores = pd.Series(scores)
    scores.index = products
    scores = scores.sort_values(ascending=False)
    to_recommend = []
    score = []
    description = []
    known_items = []

    appended = []
    for i,ii in zip(scores.index, scores):
        if len(appended)<15:
            if i not in known_items:
                to_recommend.append(i)
                score.append(ii)
                description.append(data_clean.loc[data_clean['StockCode'].isin([i]),'Description'].unique()[0])
                appended.append(0)



    lightRecom = pd.DataFrame()
    lightRecom['ProductID'] = to_recommend
    lightRecom['Description'] = description
    lightRecom['Score'] = score
    display(lightRecom.iloc[:10])
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    
    print ("User: " + str(user_id))
    print("Has Previosly Bought:")
    counter = 1
    for i in known_items:
        print(str(counter) + '- ' + i)
        counter+=1


    print("\n Recommended Items:")
    counter = 1
    for i in scores:
        print(str(counter) + '- ' + i)
        counter+=1
    return(lightRecom)



"""
LightFM (Da Net)

"""


""" 
@author: Aayush Agrawal
@Purpose - Re-usable code in Python 3 for Recommender systems
ML-small-dataset - https://grouplens.org/datasets/movielens/
"""


def create_interaction_matrix(df,user_col, item_col, rating_col, norm= False, threshold = None):
    '''
    Function to create an interaction matrix dataframe from transactional type interactions
    Required Input -
        - df = Pandas DataFrame containing user-item interactions
        - user_col = column name containing user's identifier
        - item_col = column name containing item's identifier
        - rating col = column name containing user feedback on interaction with a given item
        - norm (optional) = True if a normalization of ratings is needed
        - threshold (required if norm = True) = value above which the rating is favorable
    Expected output - 
        - Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
    '''
    interactions = df.groupby([user_col, item_col])[rating_col] \
            .sum().unstack().reset_index(). \
            fillna(0).set_index(user_col)
    if norm:
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    return interactions

def create_user_dict(interactions):
    '''
    Function to create a user dictionary based on their index and number in interaction dataset
    Required Input - 
        interactions - dataset create by create_interaction_matrix
    Expected Output -
        user_dict - Dictionary type output containing interaction_index as key and user_id as value
    '''
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0 
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict
    
def create_item_dict(df,id_col,name_col):
    '''
    Function to create an item dictionary based on their item_id and item name
    Required Input - 
        - df = Pandas dataframe with Item information
        - id_col = Column name containing unique identifier for an item
        - name_col = Column name containing name of the item
    Expected Output -
        item_dict = Dictionary type output containing item_id as key and item_name as value
    '''
    item_dict ={}
    for i in range(df.shape[0]):
        item_dict[(df.loc[i,id_col])] = df.loc[i,name_col]
    return item_dict

def runMF(interactions, n_components=30, loss='warp', k=15, epoch=30,n_jobs = 4):
    '''
    Function to run matrix-factorization algorithm
    Required Input -
        - interactions = dataset create by create_interaction_matrix
        - n_components = number of embeddings you want to create to define Item and user
        - loss = loss function other options are logistic, brp
        - epoch = number of epochs to run 
        - n_jobs = number of cores used for execution 
    Expected Output  -
        Model - Trained model
    '''
    x = sparse.csr_matrix(interactions.values)
    model = LightFM(no_components= n_components, loss=loss,k=k)
    model.fit(x,epochs=epoch,num_threads = n_jobs)
    return model

def sample_recommendation_user(model, interactions, user_id, user_dict, 
                               item_dict,threshold = 0,nrec_items = 10, show = True):
    '''
    Function to produce user recommendations
    Required Input - 
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - user_id = user ID for which we need to generate recommendation
        - user_dict = Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - threshold = value above which the rating is favorable in new interaction matrix
        - nrec_items = Number of output recommendation needed
    Expected Output - 
        - Prints list of items the given user has already bought
        - Prints list of N recommended items  which user hopefully will be interested in
    '''
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interactions.loc[user_id,:] \
                                 [interactions.loc[user_id,:] > threshold].index) \
								 .sort_values(ascending=False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
        print("Known Likes:")
        counter = 1
        for i in known_items:
            print(str(counter) + '- ' + i)
            counter+=1

        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
    return return_score_list
    

def sample_recommendation_item(model,interactions,item_id,user_dict,item_dict,number_of_user):
    '''
    Funnction to produce a list of top N interested users for a given item
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - item_id = item ID for which we need to generate recommended users
        - user_dict =  Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - number_of_user = Number of users needed as an output
    Expected Output -
        - user_list = List of recommended users 
    '''
    n_users, n_items = interactions.shape
    x = np.array(interactions.columns)
    scores = pd.Series(model.predict(np.arange(n_users), np.repeat(x.searchsorted(item_id),n_users)))
    user_list = list(interactions.index[scores.sort_values(ascending=False).head(number_of_user).index])
    return user_list 


def create_item_emdedding_distance_matrix(model,interactions):
    '''
    Function to create item-item distance embedding matrix
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
    Expected Output -
        - item_emdedding_distance_matrix = Pandas dataframe containing cosine distance matrix b/w items
    '''
    df_item_norm_sparse = sparse.csr_matrix(model.item_embeddings)
    similarities = cosine_similarity(df_item_norm_sparse)
    item_emdedding_distance_matrix = pd.DataFrame(similarities)
    item_emdedding_distance_matrix.columns = interactions.columns
    item_emdedding_distance_matrix.index = interactions.columns
    return item_emdedding_distance_matrix

def item_item_recommendation(item_emdedding_distance_matrix, item_id, 
                             item_dict, n_items = 10, show = True):
    '''
    Function to create item-item recommendation
    Required Input - 
        - item_emdedding_distance_matrix = Pandas dataframe containing cosine distance matrix b/w items
        - item_id  = item ID for which we need to generate recommended items
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - n_items = Number of items needed as an output
    Expected Output -
        - recommended_items = List of recommended items
    '''
    recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id,:]. \
                                  sort_values(ascending = False).head(n_items+1). \
                                  index[1:n_items+1]))
    if show == True:
        print("Item of interest :{0}".format(item_dict[item_id]))
        print("Item similar to the above item:")
        counter = 1
        for i in recommended_items:
            print(str(counter) + '- ' +  item_dict[i])
            counter+=1
    return recommended_items

