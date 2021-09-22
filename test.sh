# SR with symmetric scale factors
python test.py --test_only=True  \
               --dir_data='F:/LongguangWang/Data'  \
               --n_GPUs=1  \
               --data_test='Set5'  \
               --scale='2+1.6+1.55'  \
               --scale2='2+1.6+1.55'  \
               --resume=150

# SR with asymmetric scale factors
python test.py --test_only=True  \
               --dir_data='F:/LongguangWang/Data'  \
               --n_GPUs=1  \
               --data_test='Set5'  \
               --scale='1.5+1.5+1.6'  \
               --scale2='4+3.5+3.05'  \
               --resume=150

cmd /k