# AdaPool: A Diurnal-Adaptive Fleet Management Framework using Model-Free Deep Reinforcement Learning and Change Point Detection
 M. Haliem, V. Aggarwal, and B. Bhargava, “AdaPool: An Adaptive Model-Free Ride-Sharing Approach for Dispatching using Deep Reinforcement Learning”, Accepted and published, IEEE Transactions on Intelligent Transportation Systems, (T-ITS).

This project is in python 3.7 and tesnorflow 1.15.0

## These are the steps to generate the data files that serve as our dataset. These are already generated and provided @:
https://purr.purdue.edu/projects/ridesharing/files

## Setup
Below you will find step-by-step instructions to set up the NYC taxi simulation using 2016-05 trips for training and 2016-06 trips for evaluation.
### 1. Download OSM Data
```commandline
wget https://download.bbbike.org/osm/bbbike/NewYork/NewYork.osm.pbf -P osrm
```

### 2. Preprocess OSM Data
```commandline
cd osrm
docker run -t -v $(pwd):/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/NewYork.osm.pbf
docker run -t -v $(pwd):/data osrm/osrm-backend osrm-partition /data/NewYork.osrm
docker run -t -v $(pwd):/data osrm/osrm-backend osrm-customize /data/NewYork.osrm
```

### 3. Download Trip Data
```commandline
mkdir data
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-05.csv -P data/trip_records
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2016-05.csv -P data/trip_records
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-06.csv -P data/trip_records
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2016-06.csv -P data/trip_records
```

### 4. Build Docker image
```commandline
docker-compose build sim
```

### 5. Preprocess Trip Records
```commandline
docker-compose run --no-deps sim python src/preprocessing/preprocess_nyc_dataset.py ./data/trip_records/ --month 2016-05
docker-compose run --no-deps sim python src/preprocessing/preprocess_nyc_dataset.py ./data/trip_records/ --month 2016-06
```

### 6. Snap origins and destinations of all trips to OSM
```commandline
docker-compose run sim python src/preprocessing/snap_to_road.py ./data/trip_records/trips_2016-05.csv ./data/trip_records/mm_trips_2016-05.csv
docker-compose run sim python src/preprocessing/snap_to_road.py ./data/trip_records/trips_2016-06.csv ./data/trip_records/mm_trips_2016-06.csv
```

### 7. Create trip database for Simulation
```commandline
docker-compose run --no-deps sim python src/preprocessing/create_db.py ./data/trip_records/mm_trips_2016-06.csv
```

### 8. Prepare statistical demand profile using training dataset
```commandline
docker-compose run --no-deps sim python src/preprocessing/create_profile.py ./data/trip_records/mm_trips_2016-05.csv
```

### 9. Precompute trip time and trajectories by OSRM
```commandline
docker-compose run sim python src/preprocessing/create_tt_map.py ./data
```
The tt_map needs to be recreated when you change simulation settings such as MAX_MOVE.

### 10. Change simulation settings
You can find simulation setting files in `src/config/settings` and `src/simulator/settings`.

## © CLAN Labs, Purdue.


## Please cite the following papers if using any part of the code:

Marina Haliem, Vaneet Aggarwal, and Bharat K. Bhargava, "Adapool:  An adaptive model-free ride-sharing approach for dispatching using deep reinforcement learning", In BuildSys’20. @inproceedings{HaliemAB20, author    = {Marina Haliem andvVaneet Aggarwal andvBharat K. Bhargava}, title = {AdaPool: An Adaptive Model-Free Ride-Sharing Approach for Dispatching using Deep Reinforcement Learning}, booktitle = {BuildSys '20: The 7th {ACM} International Conference on Systems for Energy Efficient Buildings, Cities, and Transportation, Virtual Event, Japan, November 18-20, 2020}, pages = {304--305}, publisher = {{ACM}}, year = {2020}, url = {https://doi.org/10.1145/3408308.3431114}, doi = {10.1145/3408308.3431114}}

Marina Haliem, Ganapathy Mani, Vaneet Aggarwal, Bharat Bhargava, "A Distributed Model-Free Ride-Sharing Approach for Joint Matching, Pricing, and Dispatching using Deep Reinforcement Learning", Arxiv Pre-Print. @article{haliem2020distributed, title={A Distributed Model-Free Ride-Sharing Approach for Joint Matching, Pricing, and Dispatching Using Deep Reinforcement Learning}, author={Haliem, Marina and Mani, Ganapathy and Aggarwal, Vaneet and Bhargava, Bharat},  journal={IEEE Transactions on Intelligent Transportation Systems}, year={2021}, pages={1-12}, doi={10.1109/TITS.2021.3096537} }

Marina Haliem, Ganapathy Mani, Vaneet Aggarwal, Bharat Bhargava, "A Distributed Model-Free Ride-Sharing Algorithm with Pricing using Deep Reinforcement Learning", Computer Science in Cars Symposium, CSCS 2020. @inproceedings{10.1145/3385958.3430484, author = {Haliem, Marina and Mani, Ganapathy and Aggarwal, Vaneet and Bhargava, Bharat}, title = {A Distributed Model-Free Ride-Sharing Algorithm with Pricing Using Deep Reinforcement Learning}, year = {2020}, isbn = {9781450376211}, publisher = {Association for Computing Machinery}, address = {New York, NY, USA}, url = {https://doi.org/10.1145/3385958.3430484}, booktitle = {Computer Science in Cars Symposium}, articleno = {5}, numpages = {10} }

## Since this code uses codes developed in the papers below, please cite those too.

Abubakr Al-Abbasi, Arnob Ghosh, and Vaneet Aggarwal, "DeepPool: Distributed Model-free Algorithm for Ride-sharing using Deep Reinforcement Learning," IEEE Transactions on Intelligent Transportation Systems, vol. 20, no. 2, pp. 4714-4727, Dec 2019. @article{al2019deeppool, title={Deeppool: Distributed model-free algorithm for ride-sharing using deep reinforcement learning}, author={Al-Abbasi, Abubakr O and Ghosh, Arnob and Aggarwal, Vaneet}, journal={IEEE Transactions on Intelligent Transportation Systems}, volume={20}, number={12}, pages={4714--4727}, year={2019}, publisher={IEEE} }

A. Singh, A. Alabbasi, and V. Aggarwal, "A distributed model-free algorithm for multi-hop ride-sharing using deep reinforcement learning," IEEE Transactions on Intelligent Transportation Systems, Oct 2019 (also in NeurIPS Workshop 2019). @ARTICLE{9477304, author={Singh, Ashutosh and Al-Abbasi, Abubakr O. and Aggarwal, Vaneet}, journal={IEEE Transactions on Intelligent Transportation Systems}, title={A Distributed Model-Free Algorithm for Multi-Hop Ride-Sharing Using Deep Reinforcement Learning}, year={2021}, pages={1-11},doi={10.1109/TITS.2021.3083740}}

T. Oda and C. Joe-Wong, "Movi: A model-free approach to dynamic fleet management," IEEE INFOCOM 2018. (Their code is available at https://github.com/misteroda/FleetAI ) @inproceedings{oda2018movi, title={MOVI: A model-free approach to dynamic fleet management}, author={Oda, Takuma and Joe-Wong, Carlee}, booktitle={IEEE INFOCOM 2018-IEEE Conference on Computer Communications}, pages={2708--2716}, year={2018}, organization={IEEE} }

