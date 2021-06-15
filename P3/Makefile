CC = mpicxx
FLAGS = -O3

all: bin/bbpar_sin_difusion bin/bbpar_con_difusion bin/bbseq
	
# Ejecutables
bin/bbpar_con_difusion: obj/bbpar_con_difusion.o obj/libbb.o
	$(CC) $(FLAGS) -o $@ $^

bin/bbpar_sin_difusion: obj/bbpar_sin_difusion.o obj/libbb.o
	$(CC) $(FLAGS) -o $@ $^

bin/bbseq: obj/bbseq.o obj/libbb.o
	$(CC) $(FLAGS) -o $@ $^

# Ficheros objeto
obj/bbpar_con_difusion.o: src/bbpar.cc
	$(CC) $(FLAGS) -DDIFUSION_COTA -o $@ -c $< -I./include/

obj/bbpar_sin_difusion.o: src/bbpar.cc
	$(CC) $(FLAGS) -o $@ -c $< -I./include/

obj/bbseq.o: src/bbseq.cc
	$(CC) $(FLAGS) -o $@ -c $< -I./include/

obj/libbb.o: src/libbb.cc
	$(CC) $(FLAGS) -o $@ -c $< -I./include/


# Limpieza
clean:
	rm obj/*.o bin/bbpar_sin_difusion bin/bbpar_con_difusion bin/bbseq
