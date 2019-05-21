'''
Processing datasets. 

@author: 
'''

import scipy.sparse as sp
import numpy as np
class dataset_loader(object):
    '''
    classdocs
    '''

    def __init__(self, path, dataset, method):
        path = path + dataset
        self.dataset = dataset
        '''
        Constructor
        '''
        self.train_items = []
        print 'loading ratings...'
        self.train_matrix = self.load_data_as_matrix(path + ".train.rating")
        print 'loading test...'
        self.test_ratings = self.load_test_as_list(path + ".test.rating")
        
        if (method == 'map-bpr'):
            self.test_matrix = self.load_data_as_matrix(path + ".test.rating")

        if (method == 'dropoutnet'):
            print 'loading pretrained user latent factors...'
            self.bpr_users = np.loadtxt(path+".bpr_users_19.txt")
            self.bpr_items = np.loadtxt(path+".bpr_items_19.txt")
                        
        self.train_items = list(set(self.train_items))
        
        if (dataset == 'eachmovies'):
            self.item_features = {}#self.load_movie_features(path + ".items.txt")
        else:
            self.item_features = {}#self.load_movielens_movie_features(path + ".items.txt")
        print 'loading positively rated items...'    
    
        self.test_positives = self.load_positive_file(path + ".test.positive")
                
        self.num_users, self.num_items = self.train_matrix.shape
            
    def load_data_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
          
        if (self.dataset == 'eachmovies'):
            num_users = 74424#max(num_users, u)
            num_items = 1649#max(num_items, i)
        elif (self.dataset == 'ml-1m'):
            num_users = 6040
            num_items = 3952
        
        # Construct matrix
        mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                #try:
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2]) 
                self.train_items.append(item) 
                #except ValueError:
                #    print arr             
                mat[user, item] = 1
                #mat[user, item] = rating
                line = f.readline()    
        return mat
        
    def load_test_as_list(self, filename):
        
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                #try:
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])  
                #except ValueError:
                #    print arr 
                ratingList.append([user, item, rating])
                line = f.readline()
        #print  "In test file, total # of items -> " + str(len(set(self.ratedItems)))
        return ratingList

        
    def load_movielens_movie_features(self, filename):
        movie_features = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("::")
                item, year, item_genres = int(arr[0]), str(arr[1]), str(arr[2]).split("|")
                year_int = year[-5:-1]
                year_int = int(year_int) - 1900
                item_genres_num = []
                item_genres_encoded = [year_int]
                
                for item_genre in item_genres:
                    item_genres_num.append(self.get_genre_num(item_genre))
                
                for t in range(1,19):
                    #if item has no genre, add nothing
                    if t in item_genres_num:
                        item_genres_encoded.append(1)
                    else:
                        item_genres_encoded.append(0)
                 
                movie_features[item-1] = item_genres_encoded
                line = f.readline()
         
        #f_map = open("ml-im.items.mapper.txt", "a") 
        if (self.dataset == 'ml-1m'):
            num_items = 3952
        elif (self.dataset == 'eachmovie'):
            num_items = 1649
        else:
            num_items = 65133
                          
        for i in range(0,num_items):
            item_genres_encoded=[]
            if i not in  movie_features.keys():
                item_genres_encoded.append(0)
                for t in range(1,19):
                    item_genres_encoded.append(0)
                movie_features[i] = item_genres_encoded
            #f_map.write(str(movie_features[i])[1:-1].replace(',', '')+"\n") 
        return movie_features
    
    
    def load_eachmovie_movie_features(self, filename):
        itemFeatures = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                thisLineArray = line.split("\t")
                year = thisLineArray[1]
                year_int = int(year) - 1922
                features = [year_int]
                for i in xrange(2, len(thisLineArray)-2):
                    features.append(int(thisLineArray[i]))
                
                #if int(thisLineArray[0]) - 1 in self.ratedItems: 
                itemFeatures[int(thisLineArray[0]) - 1] = features
                line = f.readline()
                
        #f_map = open("eachmovie.items.mapper.txt", "a")       
        for i in range(0,1649):
            item_genres_encoded = []
            if i not in  itemFeatures.keys():
                item_genres_encoded.append(0)
                for t in range(1,11):
                    item_genres_encoded.append(0)
                itemFeatures[i] = item_genres_encoded
            #f_map.write(str(itemFeatures[i])[1:-1].replace(',', '')+"\n")    
        return itemFeatures
    
    
    def load_movielens_user_features(self, filename):
        user_features = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("::")
                user, gender, age, occupation, zipcode = int(arr[0]), str(arr[1]), int(arr[2]), int(arr[3]), str(arr[4])
                gender = 0 if "F" == gender.upper() else 1
                age = self.get_age_num(age)
                zipcode = zipcode.split("\n")[0]
                zipcode = int(zipcode.split("-")[0])
                
                user_features[user - 1] = [gender, age, occupation, zipcode]
                line = f.readline()
        # print users   
        return user_features
    
    
    def load_eachmovie_user_features(self, filename):
        users = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, age, sex, zipcode = int(arr[0]), arr[1], str(arr[2]), arr[3]
                if "F" == sex.upper():
                    sex = 1
                elif "M" == sex.upper():
                    sex = 2
                else:
                    sex = 0
                age = self.get_age_num(age)
                if "" == age:
                    age = 0
                if "" == zipcode:
                    zipcode = 0
                users[user-1] = [sex, age, zipcode]
                line = f.readline()               
           
        return users
    
    def get_age_num(self, age):
        if age == 18:
            return 1
        if age == 25:
            return 2
        if age == 35:
            return 3
        if age == 45:
            return 4
        if age == 50:
            return 5
        if age == 56:
            return 6
        return 0 
    
    def get_age_yahoo(self, age):
        if age > 1 and age <= 18:
            return 1
        if age <= 25:
            return 2
        if age <= 35:
            return 3
        if age <= 45:
            return 4
        if age <= 50:
            return 5
        if age <= 56:
            return 6
        if age <= 150:
            return 7
        return 0
    
    def get_genre_num(self, genre):
        genre = genre.upper()
        if "ACTION" in genre:
            return 1
        if "ADVENTURE" in genre:
            return 2
        if "ANIMATION" in genre:
            return 3
        if "CHILDREN" in genre:
            return 4
        if "COMEDY" in genre:
            return 5
        if "CRIME" in genre:
            return 6
        if "DOCUMENTARY" in genre:
            return 7
        if "DRAMA" in genre:
            return 8
        if "FANTASY" in genre:
            return 9
        if "FILM-NOIR" in genre:
            return 10
        if "HORROR" in genre:
            return 11
        if "MUSICAL" in genre:
            return 12
        if "MYSTERY" in genre:
            return 13
        if "ROMANCE" in genre:
            return 14
        if "SCI-FI" in genre:
            return 15
        if "THRILLER" in genre:
            return 16
        if "WAR" in genre:
            return 17
        if "WESTERN" in genre:
            return 18
        return 0  # no genre
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_positive_file(self, filename):
        positiveList = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                positives = []
                for x in arr[1: ]:
                    if x != "\n":
                        positives.append(int(x))
                positiveList[int(arr[0])] = positives
                line = f.readline()
        #print len(positiveList)
        return positiveList
    
    