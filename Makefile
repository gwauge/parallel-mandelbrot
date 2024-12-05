baseline: main.cpp
	icpx -qopenmp -O3 -xHost -I include/ -DBASELINE main.cpp src/baseline.cpp -o bin/baseline

static: main.cpp
	icpx -qopenmp -O3 -xHost -I include/ -DSTATIC_MULTITHREADING main.cpp src/static_multithreading.cpp -o bin/static_multithreading

run: mandelbrot
	./mandelbrot result

clean:
	rm -f result bin/*