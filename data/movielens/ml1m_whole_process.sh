echo "# Filter data"
python filter_dataset.py --input_file="data/ML1M.data"

echo ""
echo "# Preprocess for CA-DSR"
python whole_process.py --input_file="data/ML1M_account.csv" --item_file="data/ml1m.item" --maps="data/ml1m_maps.csv" --output_train="data/ml1m_train.csv" --output_valid="data/ml1m_valid.csv" --output_test="data/ml1m_test.csv"

echo ""
echo "# process item properties"
python preprocess_properties.py --input_file="data/ML1M_account.csv" --input_properties="data/ml1m.item" --maps="data/ml1m_maps.csv" --output_file="data/ml1m_properties.csv"
