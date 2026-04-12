from mpi4py import MPI
import torch

# from multiprocessing import cpu_count
#
# print("CPU的核数为：{}".format(cpu_count()))

# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()
#
# if rank == 0:
#     msg = 'Hello, world'
#     comm.send(msg, dest=1)
# elif rank == 1:
#     s = comm.recv()
#     print("rank %d: %s" % (rank, s))
# else:
#     print("rank %d: idle" % (rank))

# command = MPI.COMM_WORLD
# rank = command.Get_rank()
#
# tensor_a = torch.tensor(1.0)
# a_cuda = tensor_a.cuda(rank)
# print('Where am I', a_cuda.device)

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

if comm_rank == 0:
    data = ['s', 't', 'w', 'x', 'j']
    print(data)
else:
    data = None
local_data = comm.scatter(data, root=0)
print('rank %d, got:' % comm_rank)
print(local_data)