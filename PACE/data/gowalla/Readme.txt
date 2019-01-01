Description
================================================================================
The data set contains the checkin information of ~330K Gowalla users as well as 
the information of the visited places. It includes four data files: 

1. spot_location.txt: this data file specifies the GPS coordinates of the
places.  Each row corresponds to exactly one place with the format:
place_id longitude latitude

2. spot_category.txt: this data file specifies the category information of the 
places. Again, each row corresponds to one place and has the format: 
place_id category

3. user_network.txt: this data file specifies the (undirected) friendships among the users.
Each row corresponds to one user and has the format: 
user_id [friend_1, friend_2, ...]

4. visited_spots.txt: this data file specifies the checkin history of the users.
Each row corresponds to one user and has the format: 
user_id [place_id_1, place_id_2, ...]


Citation
================================================================================
Please cite the following paper when using this data set:

[1] Chao Zhang, Lidan Shou, Ke Chen, Gang Chen, Yijun Bei.
Evaluating Geo-Social Influence in Location-Based Social Networks.
Proceedings of the 21st ACM International Conference on Information and Knowledge Management (CIKM), 2012.


Contact
================================================================================
Chao Zhang
Email: czhang82@illinois.edu
Homepage: http://web.engr.illinois.edu/~czhang82/

