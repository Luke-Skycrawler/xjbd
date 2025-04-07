import warp as wp
import numpy as np 


@wp.kernel
def linear(x: wp.array(dtype = float), weight: wp.array2d(dtype = float), bias: wp.array(dtype =float), output: wp.array(dtype = float), relu: bool):
    i = wp.tid()
    out_dim = weight.shape[0]
    in_dim = weight.shape[1]
    
    temp = float(0.0)
    for j in range(in_dim):
        temp += x[j] * weight[i, j]
    
    o = temp + bias[i]
    if relu:
        o = wp.max(o, 0.0)
    
    output[i] = o

@wp.kernel
def jacobian_kernel(jac_in: wp.array2d(dtype = float), weight: wp.array2d(dtype = float), forward_output: wp.array(dtype = float), jac_output: wp.array2d(dtype = float), relu: bool):
    i, j = wp.tid()
    dim = jac_in.shape[0]
    s = float(0.0)
    for k in range(dim):
        s += weight[i, k] * jac_in[k, j]
    if relu: 
        s *= wp.select(forward_output[i] > 0., 0., 1.)
    jac_output[i, j] = s

class Layer:
    def __init__(self, input_dim, output_dim, name, non_linear = "relu"):
        self.input_dim, self.output_dim, self.name = input_dim, output_dim, name
        self.relu = non_linear == "relu"

        self.input = None
        self.output = None

        self.weight = None
        self.bias = None
        
        self.jac_in = None 
        self.jac_out = None
        
class WarpEncoder:
    def __init__(self, n_modes, n_nodes, n_latent = 32):

        self.n_modes = n_modes
        self.n_latent = n_latent
        self.n_nodes = n_nodes

        self.fc0 = Layer(self.n_modes, self.n_latent, "fc0", "relu")
        self.fc1 = Layer(self.n_latent, self.n_latent, "fc1", "relu")
        self.fc2 = Layer(self.n_latent, self.n_nodes * 4, "fc2", "none")

        self.nn = self.sequential(
            self.fc0, self.fc1, self.fc2
        )


    def sequential(self, *layers):
        data = np.load("data/pq4000_12d.npz")
        n = layers[0].input_dim
        self.x = wp.zeros(n, float)
        self.intermediates = [self.x]
        j0 = wp.array(np.identity(n, float), float)
        self.jacs = [j0]

        for layer in layers:
            out_dim = layer.output_dim
            name = layer.name
            layer.weight = wp.array(data[f"{name}.weight"], float)
            layer.bias = wp.array(data[f"{name}.bias"], float)

            self.intermediates.append(wp.zeros(out_dim, float))
            layer.input = self.intermediates[-2]
            layer.output = self.intermediates[-1]

            self.jacs.append(wp.zeros((out_dim, n), float))
            layer.jac_in = self.jacs[-2]
            layer.jac_out = self.jacs[-1]
        
        self.out = self.intermediates[-1]
        self.jac_out = self.jacs[-1]
        return layers

    def forward(self, x):
        self.x.assign(x)
        with wp.ScopedTimer("mlp forward"):
            for layer in self.nn:
                wp.launch(linear, dim = layer.output_dim, inputs = [layer.input, layer.weight, layer.bias, layer.output, layer.relu])
            
        return self.out.numpy()

    def jacobian_acc(self, x):
        # self.forward(x)
        n = x.shape[0]
        for layer in self.nn:
            wp.launch(jacobian_kernel, dim = (layer.output_dim, n), inputs = [layer.jac_in, layer.weight, layer.output, layer.jac_out, layer.relu])
        return self.jac_out.numpy()

    def jacobian(self, x):
        ret = np.identity(x.shape[0])
        self.forward(x)
        for layer in self.nn:
            jac = layer.weight.numpy()
            ret = jac @ ret
            if layer.relu:
                mask = (layer.output.numpy() > 0).reshape((-1, 1))    
                ret *= mask
        return ret
            
if __name__ == "__main__":
    from nn import WarpEncoder
    wp.config.max_unroll = 0
    wp.init()
    n_modes, n_nodes = 12, 30

    encoder_wp = WarpEncoder(n_modes, n_nodes)
    xnp = np.zeros(n_modes, float)
    pr = encoder_wp.forward(xnp)
    for _ in range(10):
        with wp.ScopedTimer("jacobian"):
            # jac = encoder_wp.jacobian(xnp)
            jac = encoder_wp.jacobian_acc(xnp)
    print(jac)
    # print(pr)
    # ref = np.array(
    #     [-0.4280,  0.5305, -0.3671,  0.0390,  0.5284,  0.0074, -0.3783,  0.0381,
    #     -0.5241,  0.0020, -0.4370,  0.0370,  0.3193, -0.5413, -0.2394,  0.0372,
    #      0.3991, -0.7420,  0.2271,  0.0369, -0.1931, -0.5482, -0.2351,  0.0383,
    #     -0.4546, -0.7346,  0.2581,  0.0367,  0.3000, -0.1811, -0.0593,  0.0329,
    #      0.3729,  0.5220, -0.3924,  0.0401, -0.3849, -0.5870,  0.1217,  0.0348,
    #     -0.0035,  0.5327,  0.1813,  0.1396, -0.1870, -0.3617,  0.0513,  0.0768,
    #     -0.0016,  0.4071, -0.0013,  0.1028,  0.0868,  0.0216, -0.0399,  0.0659,
    #      0.3251, -0.3864, -0.1399,  0.0273, -0.4079,  0.0148, -0.2622,  0.0321,
    #     -0.0980,  0.2144, -0.0345,  0.0472, -0.1609, -0.5864,  0.1018,  0.1313,
    #      0.0764, -0.3882,  0.0625,  0.1245,  0.0925, -0.6236,  0.1076,  0.1419,
    #      0.0540,  0.2525, -0.0401,  0.0475, -0.0044,  0.6974,  0.0964,  0.1518,
    #      0.3263, -0.5447,  0.0985,  0.0283, -0.0040,  0.8182,  0.2595,  0.1404,
    #      0.3647,  0.0216, -0.2515,  0.0342, -0.1826,  0.0214, -0.0492,  0.0270,
    #     -0.3283,  0.2830, -0.1272,  0.0330,  0.3052,  0.3352, -0.1356,  0.0320,
    #     -0.2316, -0.2385, -0.0624,  0.0350, -0.0079, -0.1809, -0.0384,  0.0835]
    # )
    # for i in range(10):
    #     pr = encoder_wp.forward(xnp)
    # print(f"diff = {pr - ref}")
