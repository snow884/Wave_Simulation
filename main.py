
import numpy as np
import math as m
import imageio
from scipy import misc

# This is the main class encapsulating all simulation methods
class wave_simulation_AI:

    # Constructor of the class. It takes the following parameters:
    # dx_in - size of one pixel in the simulation grid
    # dt_in - time step
    # sz_x_in - height of the simulation grid / pixels
    # sz_y_in - width of the simulation grid / pixels
    # steps_in - number of steps in the simulation
    # broadcast_func_in - function defining the speed of wave propagation and the elements broadcasting the waves
    def __init__(self, dx_in, dt_in, sz_x_in, sz_y_in, steps_in, broadcast_func_in):
        self.dx = dx_in
        self.dt = dt_in
        self.sz_x = sz_x_in
        self.sz_y = sz_y_in
        self.steps = steps_in
        self.broadcast_func = broadcast_func_in

    # Inplementation of the laplace operator that is used in the wave equation
    # It takes the following parameters:
    # u_array - grid containing the displacement values
    # dx - step size to be used in the aproximation fo the second derivative
    def Laplace(self, u_array, dx):
        sz_x = u_array.shape[0]
        sz_y = u_array.shape[1]

        dx2 = np.zeros((sz_x, sz_y), float)
        dy2 = np.zeros((sz_x, sz_y), float)

        dx2[1:sz_x - 1, 1:sz_y - 1] = ((u_array[0:(sz_x - 2), 1:(sz_y - 1)] - u_array[1:(sz_x - 1),
                                                                              1:(sz_y - 1)]) / dx - (
                                       u_array[1:(sz_x - 1), 1:(sz_y - 1)] - u_array[2:sz_x, 1:(sz_y - 1)]) / dx) / dx
        dy2[1:sz_x - 1, 1:sz_y - 1] = ((u_array[1:(sz_x - 1), 0:(sz_y - 2)] - u_array[1:(sz_x - 1),
                                                                              1:(sz_y - 1)]) / dx - (
                                       u_array[1:(sz_x - 1), 1:(sz_y - 1)] - u_array[1:(sz_x - 1), 2:sz_y]) / dx) / dx

        return (dx2 + dy2)

    # A simple edge detector intended for signification of the steps in the refractive index
    # It takes the following parameters:
    # c_array - The grid contining the wave propagation speed values for every pixel
    def Edge_detect(self, c_array):

        dx = np.zeros((sz_x, sz_y), float)
        contour = np.zeros((sz_x, sz_y), float)
        dx[0:(sz_x - 1), 0:(sz_y - 1)] = (np.abs(c_array[0:(sz_x - 1), 1:(sz_y)] - c_array[1:(sz_x), 1:(sz_y)]) + np.abs(
            c_array[1:(sz_x), 0:(sz_y - 1)] - c_array[1:(sz_x), 1:(sz_y)]) > 0)
        contour[dx>0]=1

        return (contour)

    # The method that actually executes the simulation
    def run(self):

        u_array = np.zeros((self.sz_x, self.sz_y), float)
        u_array_v = np.zeros((self.sz_x, self.sz_y), float)

        arr_new_r = np.zeros((self.sz_x, self.sz_y), float)
        arr_new_g = np.zeros((self.sz_x, self.sz_y), float)
        arr_new_b = np.zeros((self.sz_x, self.sz_y), float)

        outputdata = np.zeros((self.sz_x, self.sz_y, 3), int)

        writer = imageio.get_writer('/Users/ivanskya/Documents/PycharmProjects/WaveProp/out/test.mp4', fps=30)

        for t in range(0, self.steps):

            b_el, b_el_mask, c  = self.broadcast_func(t)

            u_array[b_el_mask == 1] = b_el[b_el_mask == 1]
            u_array_v = u_array_v + np.multiply(np.square(c), self.Laplace(u_array, self.dx) * self.dt)
            u_array = u_array + u_array_v * self.dt

            arr_new = u_array
            arr_new[b_el_mask == 1] = 0

            arr_new_r = np.maximum(arr_new, 0) / np.max(np.maximum(arr_new, 0) + 1e-10)
            arr_new_g = np.minimum(arr_new, 0) / np.min(np.minimum(arr_new, 0) + 1e-10)
            arr_new_b[b_el_mask == 1] = 1

            arr_new_b[self.Edge_detect(c) == 1] = 1

            outputdata[0:self.sz_x, 0:self.sz_y, 0] = arr_new_r[0:self.sz_x, 0:self.sz_y] * 255
            outputdata[0:self.sz_x, 0:self.sz_y, 1] = arr_new_g[0:self.sz_x, 0:self.sz_y] * 255
            outputdata[0:self.sz_x, 0:self.sz_y, 2] = arr_new_b * 255
            outputdata[0, 0, 0] = 1

            writer.append_data(outputdata)

# This function was used to obtain the video https://youtu.be/uBiQsoDaGkE?t=2m44s
# The function simulates a passage of a plane wave through a piece material with high refractive index
# It is meant to be passed as the last argument of the wave_simulation_AI constructor
def broadcast_func_lin_wave(t):

    broadcast_el = np.zeros((sz_x, sz_y), float)
    broadcast_el_mask = np.zeros((sz_x, sz_y), int)

    broadcast_el[0:5, 0:1024] = 0
    broadcast_el_mask[0:5, 0:1024] = 1
    broadcast_el[5:15, 5:(1024-5)] = m.sin(2*m.pi*t*0.008)
    broadcast_el_mask[5:15, 5:(1024-5)] = 1

    c_const = 6
    c = np.ones((sz_x, sz_y), float) * c_const
    c[502:542, (200):(1024-200)] = c_const * 0.5

    return broadcast_el, broadcast_el_mask, c

# This function simulates a passage of a plane wave through a prism
# It is meant to be passed as the last argument of the wave_simulation_AI constructor
def broadcast_func_prism_wave(t):

    broadcast_el = np.zeros((sz_x, sz_y), float)
    broadcast_el_mask = np.zeros((sz_x, sz_y), int)

    broadcast_el[0:5, 0:1024] = 0
    broadcast_el_mask[0:5, 0:1024] = 1
    broadcast_el[5:15, 5:(1024 - 5)] = m.sin(2 * m.pi * t * 0.024)
    broadcast_el_mask[5:15, 5:(1024 - 5)] = 1

    c_const = 6
    c = np.ones((sz_x, sz_y), float) * c_const

    for y in range(200,1024 - 200):
        c[(542-round((y-201)/3)):542, y] = c_const * 0.5

    return broadcast_el, broadcast_el_mask, c

# This function simulates a passage of a plane wave through a lens
# It is meant to be passed as the last argument of the wave_simulation_AI constructor
def broadcast_func_circlular_lens(t):
    broadcast_el = np.zeros((sz_x, sz_y), float)
    broadcast_el_mask = np.zeros((sz_x, sz_y), int)

    broadcast_el[0:5, 0:1024] = 0
    broadcast_el_mask[0:5, 0:1024] = 1
    broadcast_el[5:15, 5:(1024 - 5)] = m.sin(2 * m.pi * t * 0.020)
    broadcast_el_mask[5:15, 5:(1024 - 5)] = 1

    c_const = 6
    c = np.ones((sz_x, sz_y), float) * c_const

    for y in range(100,1024 - 100):

        c[(542-round(m.pow(m.pow((1024-200)/2,2)-m.pow(-(y-100)+((1024-200)/2),2),0.5))):542, y] = c_const * 0.6

    return broadcast_el, broadcast_el_mask, c

# This function simulates a passage of a plane wave through a Fresnel lens
# It is meant to be passed as the last argument of the wave_simulation_AI constructor
def broadcast_func_fresnel(t):
    broadcast_el = np.zeros((sz_x, sz_y), float)
    broadcast_el_mask = np.zeros((sz_x, sz_y), int)

    broadcast_el[0:5, 0:1024] = 0
    broadcast_el_mask[0:5, 0:1024] = 1

    if (t<800):
        broadcast_el[5:15, 5:(1024 - 5)] = m.sin(2 * m.pi * t * 0.024)
    broadcast_el_mask[5:15, 5:(1024 - 5)] = 1

    c_const = 6
    c = np.ones((sz_x, sz_y), float) * c_const

    for y in range(100,1024 - 100):

        y_full=m.pow(m.pow((1024 - 200) / 2, 2) - m.pow(-(y - 100) + ((1024 - 200) / 2), 2), 0.5)
        my_lambda = 2*c_const * 0.5 * (1/0.024)*0.01/0.1

        c[(542-round(y_full-round(y_full/(my_lambda)-0.5)*(my_lambda))):542, y] = c_const * 0.5

    return broadcast_el, broadcast_el_mask, c

# This function simulates a passage of a plane wave through a blazed grating
# It is meant to be passed as the last argument of the wave_simulation_AI constructor
def broadcast_func_blazed_grading(t):

    broadcast_el = np.zeros((sz_x, sz_y), float)
    broadcast_el_mask = np.zeros((sz_x, sz_y), int)

    broadcast_el[0:5, 0:1024] = 0
    broadcast_el_mask[0:5, 0:1024] = 1
    broadcast_el[5:15, 5:(1024 - 5)] = m.sin(2 * m.pi * t * 0.024)
    broadcast_el_mask[5:15, 5:(1024 - 5)] = 1

    c_const = 6
    c = np.ones((sz_x, sz_y), float) * c_const

    for y in range(200,1024 - 200):
        y_full = (542 - round((y - 201) / 3))
        my_lambda = 2*c_const * 0.5 * (1 / 0.024) * 0.01 / 0.1

        c[(542-round(y_full-round(y_full/(my_lambda)-0.5)*(my_lambda))):542, y] = c_const * 0.5

    return broadcast_el, broadcast_el_mask, c

def broadcast_func_phased_antenna_streight(t):

    broadcast_el = np.zeros((sz_x, sz_y), float)
    broadcast_el_mask = np.zeros((sz_x, sz_y), int)

    c_const = 6
    c = np.ones((sz_x, sz_y), float) * c_const

    my_lambda = c_const * (1 / 0.010) * 0.01 / 0.1

    broadcast_el[0:5, 0:1024] = 0
    broadcast_el_mask[0:5, 0:1024] = 1

    n = 10

    for a in range(0, n):

        broadcast_el_mask[5:15, int(1024/2-5+my_lambda/2*(a-n/2)):int(1024/2+5+my_lambda/2*(a-n/2))] = 1
        broadcast_el[5:15, int(1024/2-5+my_lambda/2*(a-n/2)):int(1024/2+5+my_lambda/2*(a-n/2))] = m.sin(2 * m.pi * t * 0.010)

    return broadcast_el, broadcast_el_mask, c

def broadcast_func_phased_antenna_angle(t):

    broadcast_el = np.zeros((sz_x, sz_y), float)
    broadcast_el_mask = np.zeros((sz_x, sz_y), int)

    c_const = 3
    c = np.ones((sz_x, sz_y), float) * c_const

    my_lambda = c_const * (1 / 0.008) * 0.01 / 0.1

    #broadcast_el[100:105, 0:1024] = 0
    #broadcast_el_mask[100:105, 0:1024] = 1

    n = 16

    if (t<1000):
        for a in range(0, n):
            phase=2 * m.pi / 4 * a
            phase = 0
            broadcast_el_mask[int(1024/2-1):int(1024/2+1), int(1024/2-1+my_lambda/4*(a)-my_lambda/4*n/2):int(1024/2+1+my_lambda/4*(a)-my_lambda/4*n/2)] = 1
            broadcast_el[int(1024/2-1):int(1024/2+1), int(1024/2-1+my_lambda/4*(a)-my_lambda/4*n/2):int(1024/2+1+my_lambda/4*(a)-my_lambda/4*n/2)] = m.sin(2 * m.pi * (t) * 0.008 + phase)

    return broadcast_el, broadcast_el_mask, c

def broadcast_func_phased_antenna_focus(t):

    broadcast_el = np.zeros((sz_x, sz_y), float)
    broadcast_el_mask = np.zeros((sz_x, sz_y), int)

    c_const = 6
    c = np.ones((sz_x, sz_y), float) * c_const

    my_lambda = c_const * (1 / 0.008) * 0.01 / 0.1

    broadcast_el[0:5, 0:1024] = 0
    broadcast_el_mask[0:5, 0:1024] = 1

    n = 10

    if (t<1000):
        for a in range(0, n):

            phase=2 * m.pi * m.sqrt(m.pow(500,2)-m.pow(my_lambda/2*(a-n/2),2))/my_lambda

            broadcast_el_mask[5:15, int(1024/2-5+my_lambda/2*(a-n/2)):int(1024/2+5+my_lambda/2*(a-n/2))] = 1
            broadcast_el[5:15, int(1024/2-5+my_lambda/2*(a-n/2)):int(1024/2+5+my_lambda/2*(a-n/2))] = m.sin(2 * m.pi * (t) * 0.008 + phase)

    return broadcast_el, broadcast_el_mask, c

def broadcast_func_phased_antenna_focus2(t):

    broadcast_el = np.zeros((sz_x, sz_y), float)
    broadcast_el_mask = np.zeros((sz_x, sz_y), int)

    c_const = 6
    c = np.ones((sz_x, sz_y), float) * c_const

    my_lambda = c_const * (1 / 0.008) * 0.01 / 0.1

    broadcast_el[100:105, 0:1024] = 0
    broadcast_el_mask[100:105, 0:1024] = 1

    n = 10

    if (t<1000):
        for a in range(0, n):

            phase=-2 * m.pi * m.sqrt(m.pow(400,2)-m.pow(my_lambda/2*(a-n/2),2))/my_lambda

            broadcast_el_mask[105:115, int(1024/2-5+my_lambda/2*(a-n/2)):int(1024/2+5+my_lambda/2*(a-n/2))] = 1
            broadcast_el[105:115, int(1024/2-5+my_lambda/2*(a-n/2)):int(1024/2+5+my_lambda/2*(a-n/2))] = m.sin(2 * m.pi * (t) * 0.008 + phase)

    return broadcast_el, broadcast_el_mask, c

def broadcast_func_radar(t):

    broadcast_el = np.zeros((sz_x, sz_y), float)
    broadcast_el_mask = np.zeros((sz_x, sz_y), int)

    c_const = 6
    c = np.ones((sz_x, sz_y), float) * c_const

    for y in range(-20, 20):
        c[(400-y):(400+y),int(1024/2+250-m.sqrt(m.pow(40/2,2)-m.pow(y,2))):int(1024/2+250+m.sqrt(m.pow(40/2,2)-m.pow(y,2)))]=c_const/5

    my_lambda = c_const * (1 / 0.024) * 0.01 / 0.1

    broadcast_el[0:5, 0:1024] = 0
    broadcast_el_mask[0:5, 0:1024] = 1

    n = 10

    if (t<500):
        for a in range(0, n):
            phase=2 * m.pi / 4 * a

            broadcast_el_mask[5:15, int(1024/2-5+my_lambda/2*(a-n/2)):int(1024/2+5+my_lambda/2*(a-n/2))] = 1
            broadcast_el[5:15, int(1024/2-5+my_lambda/2*(a-n/2)):int(1024/2+5+my_lambda/2*(a-n/2))] = m.sin(2 * m.pi * (t) * 0.024 + phase)

    if ((t>1000)&(t<1500)):
        for a in range(0, n):
            phase=-2 * m.pi / 4 * a

            broadcast_el_mask[5:15, int(1024/2-5+my_lambda/2*(a-n/2)):int(1024/2+5+my_lambda/2*(a-n/2))] = 1
            broadcast_el[5:15, int(1024/2-5+my_lambda/2*(a-n/2)):int(1024/2+5+my_lambda/2*(a-n/2))] = m.sin(2 * m.pi * (t) * 0.024 + phase)


    return broadcast_el, broadcast_el_mask, c

waveguide_img = misc.imread('/Path_to_your_image_goes_here/kepler_telescope.bmp')

def broadcast_func_image(t):

    broadcast_el = np.zeros((sz_x, sz_y), float)
    broadcast_el_mask = np.zeros((sz_x, sz_y), int)

    broadcast_el[0:5, 0:1024] = 0
    broadcast_el_mask[0:5, 0:1024] = 1
    broadcast_el[5:15, 5:(1024 - 5)] = m.sin(2 * m.pi * t * 0.024)
    broadcast_el_mask[5:15, 5:(1024 - 5)] = 1

    c_const = 3
    c = np.ones((sz_x, sz_y), float) * c_const
    c[(waveguide_img[0:1024,0:1024,2]/255)>0.5] = c_const * 0.6

    #my_lambda = c_const * (1 / 0.008) * 0.01 / 0.1

    #broadcast_el[112:116, (66-20):(66+20)] = 0
    #broadcast_el_mask[112:116, (66-20):(66+20)] = 1

    #broadcast_el_mask[116:118, (66-5):(66+5)] = 1
    #broadcast_el[116:118, (66-5):(66+5)] = m.sin(2 * m.pi * (t) * 0.008)

    return broadcast_el, broadcast_el_mask, c


dx = 0.1
dt = 0.01
sz_x = 1024
sz_y = 1024
steps = 3000

my_sim = wave_simulation_AI(dx, dt, sz_x, sz_y, steps, broadcast_func_image)

my_sim.run()