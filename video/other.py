# example_2.py
# import daft
# from daft.functions import file
# import soundfile as sf
# import whisper

# # Load model
# model = whisper.load_model("tiny")

# @daft.func
# def transcribe(file: daft.File) -> str:
#     """Transcribes an audio file using OpenAI Whisper"""
#     with file.open() as f:
#         audio, _ = sf.read(f, dtype='float32')
#     result = model.transcribe(audio)
#     return result['text']

# # Create dataframe from all flac files in directory
# df = daft.from_glob_path("./LibriSpeech/dev-clean/**/*.flac")

# # Process all files
# df = df.select(
#     df["path"],
#     transcribe(file(df["path"])).alias("transcription")
# )

# # Write results
# df.write_csv("transcriptions.csv")

# make csv like things
# process data
# write into lakesoul

# find UCF101_subset -name "*.avi" | awk -F/ '{
#   split_name=$2;
#   label=$3;
#   file=$4;
#   video_id=file;
#   sub(/\.avi$/, "", video_id);
#   print video_id "," split_name "," label "," $0
# }' > ucf101_metadata.csv
# 
# sed -i '1i video_id,split,label,video_path' ucf101_metadata.csv
# 
# # ffprobe -v error \
#   -select_streams v:0 \
#   -show_entries stream=width,height,r_frame_rate,codec_name \
#   -show_entries format=duration \
#   -of csv=p=0 \
#   UCF101_subset/train/BaseballPitch/v_BaseballPitch_g01_c04.avi
#
#ffmpeg -y -i UCF101_subset/train/BaseballPitch/v_BaseballPitch_g01_c04.avi \
#   -frames:v 1 thumbnail.jpg

# 或者抽第 1 秒附近的一帧：

# ffmpeg -y -ss 1 -i UCF101_subset/train/BaseballPitch/v_BaseballPitch_g01_c04.avi \
#   -frames:v 1 thumbnail.jpg

# 接下来你就可以把 metadata 扩成：

# video_id,split,label,video_path,codec,width,height,fps,duration_sec,thumbnail_path 