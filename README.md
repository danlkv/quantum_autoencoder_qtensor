## Quantum autoencoder


We have an encoder circut that encodes a particular state. The trick is that we measure some qubits that we call "Trash qubits", and we want our encoder circuit to produce those states that have zeroes on those trash qubits. This way if we apply a reversed autoencoder circuit, we will obtain the original quantum state at the output.

To do this, we have two options:

1. Minimize sum probabilities of states that have non-zero values on the trash qubits
2. Maximize sum of probabilities of sates that have zeroes on the trash qubits

It is easier to do the 2nd option.


### Implementation using qtensor

```python 
import qtensor
from qtensor.contraction_backends import TorchBackend

N = 10
qubits = range(N)
trash_qubits = range(8)
useful_qubits = [q for q in qubits if q not in trash_qubits]

def loss(params):
    # The gen_autoencoder_circuit should use qtensor.TorchBuilder for gate creation
    cirucit = gen_autoencoder_circuit(params, N)
    sim = qtensor.QtreeSimulator(backend=TorchBackend())
    amps = sim.simulate_batch(cirucit, batch_vars=useful_qubits)

    prob_sum = np.sum(np.abs(
            np.square(amps)
        ))
    return - prob_sum
```


### Getting rid of density matrix?

Let's say we have 4 qubits, last 2 qubits are 'thash' qubits.

After optimizaiton we hope to get psi to be:
```
0000 - large apmiltude
0001 - small amplitude
0010 - small amplitude
0011
0100 - large amplitude
...

```

Now we want to get the compressed state.

There are two options:

1. Calculate density matrix from the psi, `rho = |psi><psi|`. Then trace thrash quibts to get reduced density matrix. `rho_c = \sum_{thrash} <thrash|psi><psi|thrash>`.
2. Sum *the state* over the thrash qubits. The compressed state is `phi` equals `phi_ij = \sum_{kl} psi_{ijkl}`. Effectively this means that to get amplitude `xx` of `phi`, you have to sum over `xx00, xx01, xx10, xx11`

### TODO:

1. Add YPhase gate to `TorchFactory`
2. Add posssibility to specify gate factory for qiskit importer
3. Correct signotures of circut generating functions
            
