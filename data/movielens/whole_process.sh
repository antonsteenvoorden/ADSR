echo "# Filter data"
python filter_dataset.py

# echo ""
# echo "# Add time context"
# python add_time_context.py

echo ""
echo "# Preprocess for CA-DSR"
python whole_process.py

echo ""
echo "# process item properties"
python preprocess_properties.py

# echo ""
# echo "# Preprocess for CA-DSR time variants"
# python preprocess_ratings_with_time.py

# echo ""
# echo "# Making item embeddings for diversity measure"
# python embed_items_with_mf.py