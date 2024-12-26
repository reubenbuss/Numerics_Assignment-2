import time
import time_v3 as t3

start = time.time()

'''
Running the code as is takes around 90 seconds.

To produce the convergence plots takes around 30mins 
so it is commented out and I would just take my work for it 
as it might not even work as I've not checked it since I wrote it.
'''

# change res if you want different values for the tables
t3.numbers_from_system(btype=1, res=5)

# model = 1 (Backwards Euler) frame at 20,000s with boundary_type = 1 D = 1000 for minimal diffusion
t3.plot_specific_frames_from_scratch(
    frames=[20000], btype=1, res=5, D=1000, model=1, concentration_at_reading=False)
# model = 4 (Crank-nicholson) frame at 20,000s with boundary type = 1 D=0 for stability as dt=10
t3.plot_specific_frames_from_scratch(
    frames=[20000], btype=5, res=5, D=0, model=4, concentration_at_reading=False)
# Crank-Nicholson with inflow outflow boundary
t3.plot_specific_frames_from_scratch(
    frames=[15000, 16000], btype=1, res=5, D=0, model=4, concentration_at_reading=False)


# Recommended
# make a nice gif of the best simulation I have, plus bonus plot of the concentration of Psi over reading
t3.make_gif_from_scratch(btype=1, res=5, D=10000,
                         model=1, concentration_at_reading=True)

# Not recommended
# If you're feeling very brave and have 30mins spare you can create the convergence plot with this code
# t3.save_psi(res=40)
# t3.save_psi(res=20)
# t3.save_psi(res=10)
# t3.save_psi(res=5)
# t3.save_psi(res=2.5)
# t3.save_psi(res=1.25)
# t3.convergence_test()

end = time.time()
print(f'Completion time {end-start}')
