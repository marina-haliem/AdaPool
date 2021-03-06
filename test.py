import multiprocessing

def greedy_wrapper(chunks):
	numCpu = multiprocessing.cpu_count()
	# p = multiprocessing.Pool(processes=5)
	# chunks = list()
	n = 0
	# while n < 5:
	with multiprocessing.Pool(5) as p:
		narrow, commands = zip(*p.map(greedy_insertion, chunks, 2))
		p.close()
		n += 1

	# p.join()
	# p.terminate()
	return narrow, commands

def greedy_insertion(chuncks):
	print("Here")
	sum_id = 0
	sum_cust = 0
	print(chuncks)
	print(multiprocessing.current_process())
	# for c in chuncks:
	# 	print(c)
		# sum_id += c["id"]
		# sum_cust += c["cust"]
	print("There")
	return sum_id, sum_cust

if __name__ == '__main__':
	numCpu = multiprocessing.cpu_count()
	items = [{"id": 1, "cust": 2}, {"id": 2, "cust": 2}, {"id": 3, "cust": 2}, {"id": 4, "cust": 2},
			  {"id": 5, "cust": 2},
			  {"id": 6, "cust": 2},
			  {"id": 7, "cust": 2}, {"id": 8, "cust": 2}, {"id": 9, "cust": 2}, {"id": 10, "cust": 2}]

	items2 = [1,2,3,4,5,6,7,8,9,10]

	chunksize = int((len(items) + numCpu) / numCpu)
	# print(chunksize)
	chunks = []
	for i in range(0, len(items), chunksize):
		if i + chunksize >= len(items):
			chunks.append(items[i:len(items)])
		else:
			chunks.append(items[i:i + chunksize])
	# # print(chunks)

	id, cust = greedy_wrapper(items2)

	print(id, cust)