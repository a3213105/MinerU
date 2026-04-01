OUTPUT=$1
SOURCE=$2 
python -m mineru.cli.models_download -m pipeline -s $SOURCE -o $OUTPUT
python -m mineru.main --init
