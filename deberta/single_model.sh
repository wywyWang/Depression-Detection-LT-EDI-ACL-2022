current_time=`date +%Y%m%d-%H%M%S`
model_path="./model/${current_time}_"$1"/"
python train.py --output_folder_name ${model_path}
# python evaluate.py ${model_path}