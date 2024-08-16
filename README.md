## Title: An Efficient ML-based Detection Framework for DDoS Attacks in 5G Near Real-Time RIC
### Abstract:
As 5G networks evolve, detecting Distributed Denial of Service (DDoS) attacks remains crucial. This paper presents a supervised learning ML-based 5G Core DDoS detection framework, that can be implemented as an xApp within the Near-RT RIC. Utilising statistical features from UE and CELL-specific key performance metrics, our framework achieves high detection rates and low false positives. Testing confirms its effectiveness, with F1-scores of up to 98\%, and detection rates of over 91\%. These results demonstrate the potential use of ML techniques in Open RAN environments to defend 5G networks.

### Open RAN based architecture:
![plot](https://github.com/sotirischatzimiltis/DDoS_5G_network_KPM/blob/main/Figrures/arch_working_final_letter.png)

### Dataset
The dataset can be downloaded from here: [NCSRD-DS-5GDDoS](https://zenodo.org/records/10671494)

### Data Pre-processing
1. UE feature selection [script](https://github.com/sotirischatzimiltis/DDoS_5G_network_KPM/blob/main/Scripts/ue_data_feature_selection.ipynb)
2. Cell feature selection [script](https://github.com/sotirischatzimiltis/DDoS_5G_network_KPM/blob/main/Scripts/cell_level_feature_selection.ipynb)
3. Merge UE and Cell data [script](https://github.com/sotirischatzimiltis/DDoS_5G_network_KPM/blob/main/Scripts/merge_dataset.ipynb)
   
### ML training
1. GridSearch for traditional ml models [script](https://github.com/sotirischatzimiltis/DDoS_5G_network_KPM/blob/main/Scripts/gridsearch_cv.ipynb)
1. Download and execute [supervised ml training script](https://github.com/sotirischatzimiltis/DDoS_5G_network_KPM/blob/main/Scripts/supervised_learning_approach.py)
2. For LSTM training you can download the following script [lstm](https://github.com/sotirischatzimiltis/DDoS_5G_network_KPM/blob/main/Scripts/lstm.ipynb)
   > Note: [auxiliary.py](https://github.com/sotirischatzimiltis/DDoS_5G_network_KPM/blob/main/Scripts/auxiliary.py) script should be downloaded. Responsible to form the training and inference sequences
