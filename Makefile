mandelbrot: mandelbrot.cpp
	icpx -qopenmp -O3 -xHost mandelbrot.cpp -o mandelbrot

run: mandelbrot
	./mandelbrot result

clean:
	rm -f mandelbrot result