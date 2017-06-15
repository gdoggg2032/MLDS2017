# [S2S, RL, BEST] [INPUT_FILE] [OUTPUT_FILE] 


MODEL=$1
INPUT_FILE=$2
OUTPUT_FILE=$3

if [ $MODEL = "S2S" ]; then
	cat model_chatbot_seq2seq.zip.* > model_chatbot_seq2seq.zip
	unzip model_chatbot_seq2seq.zip
	python3 chatbot.py --mode 1 --log model_chatbot_seq2seq --test_file_path $INPUT_FILE --test_output_path $OUTPUT_FILE
elif [ $MODEL = "RL" ]; then
	cat model_chatbot_rl.zip.* > model_chatbot_rl.zip
	unzip model_chatbot_rl.zip
	python3 chatbot.py --mode 1 --log model_chatbot_rl --test_file_path $INPUT_FILE --test_output_path $OUTPUT_FILE
elif [ $MODEL = "BEST" ]; then
	cat model_chatbot_rl.zip.* > model_chatbot_rl.zip
	unzip model_chatbot_rl.zip
	python3 chatbot.py --mode 1 --log model_chatbot_rl --test_file_path $INPUT_FILE --test_output_path $OUTPUT_FILE
fi

