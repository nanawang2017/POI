This is the instruction for using RankGeoFM.


-----------------------------------------------------------------------------

1. Data Preparation:


- Firstly, you need to create two folders. One for input data, named $dataset_name$ (e.g. 'SG'), and the other is for intermediate results, named $dataset_name_$_POI_rec_Iterations (e.g. 'SG_POI_rec_Iterations'). 


- Secondly, you need to put your data in input data folder. See the examples in ./SG for the data format.
	|---- ./SG
	|	|---- tensor_lat_lng.txt 			// GPS coordinate of POIs.
	|	|
	|	|---- tensor_size.dat 				// Data size. (note: better add ',24' even it is not for time-aware POI recommendation.)
	|	|
	|	|---- train_tensor.dat 				// Training data.
	|	|
	|	|---- tune_tensor.dat 				// Tuning data.
	|	|
	|	|---- test_tensor.dat 				// Test data.


- Thirdly, run ./getDist.m like 'getDist(filename, nk)', to pre-obtain the nearest neighbors of POIs. 

	Parameters:
		filename: str
			Dataset name. 
		nk: int 
			Number of nearest neighbors you want to have for each POI.

	Example: getDist('SG', 300)

-----------------------------------------------------------------------------

2. Model Training:

- After data preparation, run ./RankGeoFM.m like 'RankGeoFM(filename, K, k_1, connect)'.

	Parameters:
		filename: str
			Dataset name.
		K: int
			Number of dimension of latent matrices U and L.
		k_1: int
			Number of nearest neighbors for each POI.
			note: k_1 <= nk should be satisfied.
		connect: 0/1
			This value could be either 0 or 1. If the training is interrupted, you can set connect as 1 and run 'RankGeoFM(filename, K, k_1, 1)',
			so that the training will be continued. For training a new model, just set it as 0.

	Example: RankGeoFM('SG', 100, 100, 0)
 
 - For other parameters in the paper, please check ./RankGeoFM.m.

-----------------------------------------------------------------------------

3. Evaluation:

- If you want to compute precision&recall, run ./test_performance_2new.m.

	Parameters:
		filename: str
			Dataset name.
		flagTest: 0/1
			To evaluate the model on tuning/test data.
		N: list
			Specify precision/recall at top-N items.
		flagMethod: int
			It should always be 4.

	Example: test_performance_2new('SG', 1, [5 10 20 30 50], 4)

- If you want to obtain the recommended POI ids, run ./test_performance_3new.m. The recommended POIs of user i will be stored at the ith line of ./result/sigir15li.dat

	Parameters:
		filename: str
			Dataset name.
		top_k: int
			Specify top k recommended POI ids to be stored.

	Example: test_performance_3new('SG', 100)

-----------------------------------------------------------------------------

Welcome to send me an email if you have any question or suggestion. Thank you very much.

E-mail: liuy0130@e.ntu.edu.sg

Yiding Liu  -  Jan 21, 2016


