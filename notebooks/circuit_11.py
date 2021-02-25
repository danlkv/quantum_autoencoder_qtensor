import numpy as np
pi = np.pi

def circuit_11(layers, composer, parameters=None):
    ops = composer.operators
    L = composer.n_qubits
    if L%2:
        raise Exception("Number of qubits should be even")
    q = composer.qubits
    if parameters is None:
        a=2*pi*np.random.random(layers*(4*L-4))
    else:
        a = parameters
    for layer in range(layers):
        for i in range(L):
            ##adds first layer of rotations
            composer.apply_gate(ops.YPhase, q[i], alpha=a[layer*(4*L-4)+i])
            composer.apply_gate(ops.ZPhase, q[i], alpha=a[layer*(4*L-4)+i+L])
        #adds first layer of CNOTs
        for j in range(0, L, 2):
            composer.apply_gate(ops.cX, q[j+1], q[j])
        ##add second layer of rotations
        for k in range(0, L-2):
            composer.apply_gate(ops.YPhase, q[k+1], alpha=a[layer*(4*L-4)+2*L+k])
            composer.apply_gate(ops.ZPhase, q[k+1], alpha=a[layer*(4*L-4)+2*L+L-2+k])
        ##add second layer of CNOTs
        for l in range(1,L-1,2):
            composer.apply_gate(ops.cX, q[l+1], q[l])

    return composer.circuit
