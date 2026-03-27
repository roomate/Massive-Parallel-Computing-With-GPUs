# Variables
CC = nvcc
CFLAGS = -g -arch=compute_60 -code=sm_75
SOURCES = main.cu MC.cu wrappers.cu utils.cu 
OBJECTS = $(SOURCES:.cu=.o)
TARGET = MC

# Règle par défaut: Si une des dépendances a changé, la target correspondante sera exécuté.
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^

# Compilation des fichiers .cu en .o
%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

# Déclaration des cibles phony
.PHONY: all clean fclean re

# Nettoyage des fichiers objets
clean:
	rm -f $(OBJECTS)

# Nettoyage complet (fichiers objets et exécutable)
fclean: clean
	rm -f $(TARGET)

# Refaire la compilation
re: fclean all

