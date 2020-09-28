echo 'Before starting, please download'
echo 'Word2Vec Pretrained Vectors - GoogleNews-vectors-negative300.bin'
echo ''
echo 'Restructuing Training data'
python3 restructure_training_data.py
echo 'Converting Training data to vectors'
python3 to_feature_vectors.py
echo 'Building Distractor Pool'
python3 make_distractor_pool.py
echo 'Training and creating models'
python3 train.py
echo 'Testing ... (This will take a while)'
python3 test.py
echo 'Formatting Prediction file for submission'
python3 formatting_predictions.py
