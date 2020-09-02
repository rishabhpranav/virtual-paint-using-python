from multiprocessing import Process, Pipe
import integration
import cv

if __name__ == '__main__':
	# Creating pipes to send data between two processes
    parent_conn, child_conn = Pipe()

    # GUI component receives data
    p1 = Process(target=integration.main, args=(parent_conn,))

    # CV component sends data
    p2 = Process(target=cv.main, args=(child_conn,))

	# Sets up both the compupter vision components and GUI windows to run in the Pi.
    p1.start()
    p2.start()
    p1.join()
    p2.join()