all: prepare reneel.out run

# argument 1: Positive Integer, parameter for Randomized Greedy (usually 2)
# argument 2: Positive Integer, maximum and initial ensemble size of partitions used in RenEEL
# argument 3: Positive Integer, ensemble size of partitions of the reduced network for iteration part in RenEEL
run: reneel.out
	./reneel.out 2 10 5

reneel.out: main.c help.c rg.c
	gcc main.c help.c rg.c -lm -o reneel.out

prepare: work.sh
	./work.sh karate.txt

clean:
	rm reneel.out
	find ./ -name "*.txt" -not -name "karate.txt" -exec rm {} \;

