CXXFLAGS = $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') $(shell pkg-config --cflags pinocchio lapack)
LDFLAGS = $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') $(shell pkg-config --libs pinocchio lapack)

all: dynamics_op.so

dynamics_op.so: op.o kernel-loss.o kernel-loss-gradient.o
	$(CXX) -std=c++14 -shared -o $@ $^ $(LDFLAGS)

%.o: %.cc
	$(CXX) -std=c++14 -fPIC -Wall -Wextra -Wno-deprecated-declarations -Wno-sign-compare -Wno-unused-parameter -O2 -DNDEBUG -c -o $@ $< $(CXXFLAGS)

clean:
	@$(RM) dynamics_op.so op.o kernel-loss.o kernel-loss-gradient.o

.PHONY: all clean
