# Deep Stacked Ensemble Recommender

Rasaq Otunba, Raimi A. Rufai, and Jessica Lin. 2019. Deep Stacked EnsembleRecommender. InProceedings of 31st International Conference on Scientificand Statistical Database Management, Santa Cruz, CA, USA, July 23–25, 2019. 

4 recommender systems implemented: DSER, NeuMF, BPR, ITEMPOP

## Example to run the codes.
The instruction of commands has been clearly stated in the codes (see the  parse_args function). 

Run DSER
```
python main.py --method dser --dataset ml-1m --epochs 50 --lr 0.005
```


### Dataset
We provide 3 processed sets of datasets used in the paper: MovieLens, Pinterest and Book-Crossing in the data folder

#.train.rating: 
- Train file.
- Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

test.positive
- Test file (positive instances).
- Each line corresponds to the line of test.rating, containing all positive items.  
- Each line is in the format: (userID,itemID)\t positiveItemID1\t positiveItemID2 ...

Versions of dependencies that worked can be found in the requirements.txt file

Last Update Date: May 20, 2019