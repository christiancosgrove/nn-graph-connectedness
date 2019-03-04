REPEATS=5
EPOCHS=1000
INPUT='./data/graphs6.txt'
OUTPUT='train_graphs6.txt'


# rm train_graphs5.txt

for ex in $(seq 2 2 9; seq 10 20 100; seq 100 50 300; seq 300 100 1000)
do
	for i in $(seq 0 $REPEATS)
	do	
		echo $ex
		python gen_data.py --train_examples $ex --epochs $EPOCHS --data $INPUT >> $OUTPUT
	done
done