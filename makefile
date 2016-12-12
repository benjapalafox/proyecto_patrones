all: proj

proj:
	mpicc -o proyecto proyecto.c -Wall -lm

run: proj
	mpirun -n 8 ./proyecto
clean:
	rm -rf ./training
	rm -rf ./testing
	rm -rf ./errors
	rm -rf mnist_results.txt
	mkdir training
	mkdir testing
	mkdir errors

