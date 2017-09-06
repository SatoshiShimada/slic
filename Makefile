DEBUG=0

CXX=g++
CFLAGS=-std=c++14 -Wall `pkg-config --cflags opencv`
LIBS=-lm `pkg-config --libs opencv`
OPTS=

EXE=slic
OBJ=slic.o
OBJDIR=./obj/
VPATH=./src/

ifeq ($(DEBUG), 1)
OPTS+=-O0 -g
endif

OBJS=$(addprefix $(OBJDIR), $(OBJ))

all: obj $(EXE)

$(EXE): $(OBJS)
	$(CXX) $(OPTS) $(CFLAGS) -o $@ $< $(LIBS)

$(OBJDIR)%.o: %.cpp
	$(CXX) $(OPTS) $(CFLAGS) -o $@ -c $<

obj:
	mkdir -p ./obj

clean:
	rm $(OBJS) $(EXE)
.PHONY: clean

