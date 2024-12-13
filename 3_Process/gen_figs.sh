source ../create_env.sh

python3 referenced_examples.py --dataset_directory ./gen_figures/ --cache_only && \
python3 single_frame.py --cache_only && \
python3 num_frames_videos.py --cache_only && \
python3 confidence_cdf.py --cache_only && \
python3 multi_frame.py --cache_only