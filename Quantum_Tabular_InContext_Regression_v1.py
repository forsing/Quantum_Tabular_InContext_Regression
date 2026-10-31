# -*- coding: utf-8 -*-

# Quantum_Tabular_InContext_Regression


"""
QUANTUM TABULAR IN-CONTEXT REGRESSION V1 — Loto 7/39

Adaptacija principa predvidjanja iz konteksta.

Ovo nije originalni TabPFN Transformer niti unapred obucen
foundation model.

Model koristi stvarne hronoloske zadatke:
    poznati raniji prelazi -> naredna kombinacija.

Kontekst:
kompleksna memorija uparenih istorijskih prelaza,
predstavljena centriranim unakrsnim momentima amplituda.

Memorija koristi sve ranije poznate prelaze.
Ne koristi samo poslednji prozor istorije.

Qiskit:
1. nauceno kolo kvantnog upita;
2. kontekstualni amplitudni filter;
3. nauceno izlazno kvantno kolo.

U treningu se filter primenjuje analiticki radi efikasnosti.
Kvantni upit i izlaz racunaju se Qiskit evolucijom.

Samoprovere i NEXT dodatno proveravaju isti filter
eksplicitnim Qiskit kolom sa pomocnim kubitom.

Sedam identicnih, nezavisno pripremljenih izlaznih registara
ima po sest kubita. Zbog istog ulaza i parametara,
njihov zajednicki pojedinacni izlaz racuna se jednom.

Zajednicka raspodela:
    P(S | upit, kontekst) = product_{j in S} q_j / e_7(q)

Uslov:
sedam razlicitih brojeva od 1 do 39.

Nema dodatnog naucenog sprezanja izmedju buducih brojeva,
osim zajednickog uslova kardinalnosti.

Nema klasifikatora ni klasicnog prediktivnog regresora.
Klasicni deo priprema kontekst, racuna izvode,
normalizaciju i optimizaciju.

Parametri se ne optimizuju ponovo za svaki upit.
Promena konteksta menja amplitude uz iste naucene uglove.
"""


import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import csv
import hashlib
import math
import random
import sys
import time
import traceback

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import scipy
import qiskit

from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZZGate
from qiskit.quantum_info import Statevector

random.seed(39)
np.random.seed(39)


VERSION = "QUANTUM_TABULAR_INCONTEXT_REGRESSION_V1_20261031"


FILES = (
    Path("/data/loto7_4712_k87.csv"),
    Path("/data/loto7_4712_k87_loto_2978.csv"),
    Path("/data/loto7_4712_k87_loto_plus_1734.csv"),
)


NUMBERS = 39
PICKS = 7

QUBITS = 6
DIMENSION = 1 << QUBITS
LAYERS_PER_BLOCK = 1

CONTEXT_STRENGTHS = (0.20, 0.45)

RIDGE = 0.01
ANGLE_BOUND = math.pi

MAX_ITERATIONS = 48
MAX_EVALUATIONS = 80

GRADIENT_LIMIT = 5.0e-5
VALIDATION_TIE = 1.0e-5

MEMORY_EPSILON = 1.0e-12

if "__file__" in globals():
    SCRIPT_PATH = Path(__file__).resolve()
    SCRIPT_DIR = SCRIPT_PATH.parent
    CODE_HASH = hashlib.sha256(
        SCRIPT_PATH.read_bytes()
    ).hexdigest()
else:
    SCRIPT_PATH = None
    SCRIPT_DIR = Path.cwd()
    CODE_HASH = hashlib.sha256(
        VERSION.encode("utf-8")
    ).hexdigest()

CACHE_DIR = (
    SCRIPT_DIR
    / "Quantum_Tabular_InContext_Regression_V1_checkpoints"
)


def say(*values):
    print(*values, flush=True)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def finite(values, label):
    require(
        np.all(np.isfinite(values)),
        f"{label}: nekonacan rezultat.",
    )


def next_power_of_two(number):
    require(number >= 1, "Broj uzoraka mora biti pozitivan.")
    return 1 << (int(number) - 1).bit_length()


def digest_items(*items):
    digest = hashlib.sha256()

    for item in items:
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)

            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(memoryview(array).cast("B"))
        else:
            digest.update(repr(item).encode("utf-8"))

    return digest.hexdigest()


def atomic_npz(path, **values):
    temporary = path.with_name(
        path.name + f".{os.getpid()}.tmp"
    )

    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


def maximum_run(numbers):
    longest = 1
    current = 1

    for previous, value in zip(numbers[:-1], numbers[1:]):
        if value == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


class FixedSizeDistribution:
    def __init__(self, numbers, picks):
        self.numbers = int(numbers)
        self.picks = int(picks)

        require(
            1 <= self.picks <= self.numbers,
            "Neispravna velicina kombinacije.",
        )

    def statistics(self, log_weights):
        x = np.asarray(log_weights, dtype=np.float64)

        require(
            x.ndim == 2
            and x.shape[1] == self.numbers
            and len(x) > 0,
            "Neispravan oblik tezina raspodele.",
        )

        finite(x, "Logaritamske tezine")

        n = len(x)
        m = self.numbers
        k = self.picks

        prefix = np.full(
            (n, m + 1, k + 1),
            -np.inf,
            dtype=np.float64,
        )

        suffix = np.full_like(prefix, -np.inf)

        prefix[:, 0, 0] = 0.0
        suffix[:, m, 0] = 0.0

        for j in range(m):
            prefix[:, j + 1, :] = prefix[:, j, :]

            prefix[:, j + 1, 1:] = np.logaddexp(
                prefix[:, j, 1:],
                x[:, j, None] + prefix[:, j, :k],
            )

        for j in range(m - 1, -1, -1):
            suffix[:, j, :] = suffix[:, j + 1, :]

            suffix[:, j, 1:] = np.logaddexp(
                suffix[:, j + 1, 1:],
                x[:, j, None] + suffix[:, j + 1, :k],
            )

        log_z = prefix[:, m, k]

        finite(log_z, "Normalizacija")

        require(
            np.max(
                np.abs(log_z - suffix[:, 0, k])
            ) < 2.0e-9,
            "Prednja i zadnja normalizacija se ne slazu.",
        )

        marginal = np.empty_like(x)

        for j in range(m):
            leave_one = np.logaddexp.reduce(
                prefix[:, j, :k]
                + suffix[:, j + 1, k - 1::-1],
                axis=1,
            )

            marginal[:, j] = np.exp(
                x[:, j] + leave_one - log_z
            )

        finite(marginal, "Marginalne verovatnoce")

        require(
            np.min(marginal) >= 0.0
            and np.max(marginal) <= 1.0 + 2.0e-8,
            "Marginalne verovatnoce su izvan opsega.",
        )

        require(
            np.max(
                np.abs(
                    marginal.sum(axis=1) - self.picks
                )
            ) < 2.0e-7,
            "Zbir marginalnih verovatnoca nije ispravan.",
        )

        return log_z, marginal

    def log_probabilities(self, log_weights, targets):
        targets = np.asarray(targets, dtype=np.int64)

        require(
            targets.shape
            == (len(log_weights), self.picks)
            and np.all(targets >= 0)
            and np.all(targets < self.numbers)
            and np.all(np.diff(targets, axis=1) > 0),
            "Neispravne ciljne kombinacije.",
        )

        log_z = self.statistics(log_weights)[0]

        result = (
            np.take_along_axis(
                log_weights, targets, axis=1
            ).sum(axis=1)
            - log_z
        )

        finite(result, "Logaritmi verovatnoca kombinacija")

        require(
            np.max(result) <= 2.0e-8,
            "Verovatnoca kombinacije prelazi jedan.",
        )

        return result

    def maximum(self, log_weights):
        values = np.asarray(
            log_weights, dtype=np.float64
        )

        require(
            values.shape == (self.numbers,),
            "Neispravan oblik za trazenje maksimuma.",
        )

        finite(values, "Tezine maksimuma")

        selected = np.sort(
            np.argsort(
                -values, kind="stable"
            )[:self.picks]
        )

        log_z = self.statistics(
            values[None, :]
        )[0][0]

        log_probability = float(
            values[selected].sum() - log_z
        )

        require(
            -math.log(
                math.comb(self.numbers, self.picks)
            ) - 2.0e-8
            <= log_probability
            <= 2.0e-8,
            "Neispravna verovatnoca maksimuma.",
        )

        return selected, log_probability

    def independent_log_z(self, log_weights):
        values = np.asarray(
            log_weights, dtype=np.longdouble
        )

        shift = np.max(values)
        weights = np.exp(values - shift)

        coefficients = np.zeros(
            self.picks + 1,
            dtype=np.longdouble,
        )

        coefficients[0] = 1.0

        for weight in weights:
            for degree in range(self.picks, 0, -1):
                coefficients[degree] += (
                    weight * coefficients[degree - 1]
                )

        require(
            coefficients[self.picks] > 0.0,
            "Nezavisna normalizacija nije pozitivna.",
        )

        return float(
            np.log(coefficients[self.picks])
            + self.picks * shift
        )


DISTRIBUTION = FixedSizeDistribution(NUMBERS, PICKS)


class QuantumBatch:
    def __init__(self, inputs):
        self.inputs = np.asarray(
            inputs, dtype=np.complex128
        )

        require(
            self.inputs.ndim == 2
            and len(self.inputs) > 0,
            "Neispravan paket amplituda.",
        )

        self.samples, self.dimension = self.inputs.shape

        require(
            self.dimension & (self.dimension - 1) == 0,
            "Dimenzija nije stepen dvojke.",
        )

        finite(self.inputs, "Ulazne amplitude")

        require(
            np.max(
                np.abs(
                    np.sum(
                        np.abs(self.inputs) ** 2,
                        axis=1,
                    ) - 1.0
                )
            ) < 2.0e-10,
            "Ulazne amplitude nisu normalizovane.",
        )

        self.padded = next_power_of_two(
            self.samples
        )

        matrix = np.zeros(
            (self.padded, self.dimension),
            dtype=np.complex128,
        )

        matrix[:self.samples] = (
            self.inputs / math.sqrt(self.samples)
        )

        self.packed = matrix.reshape(-1)


def quantum_gate(name, angle):
    if name == "rx":
        return RXGate(float(angle))

    if name == "ry":
        return RYGate(float(angle))

    if name == "rzz":
        return RZZGate(float(angle))

    raise RuntimeError("Nepoznata kapija.")


class QuantumBlock:
    def __init__(
        self,
        qubits=QUBITS,
        layers=LAYERS_PER_BLOCK,
    ):
        self.qubits = int(qubits)
        self.layers = int(layers)

        require(
            self.qubits >= 2 and self.layers >= 1,
            "Neispravan kvantni blok.",
        )

        self.dimension = 1 << self.qubits
        self.size = 3 * self.qubits * self.layers

        self.indices = np.arange(
            self.dimension, dtype=np.int64
        )

    def operations(self, parameters):
        values = np.asarray(parameters).reshape(
            self.layers, 3, self.qubits
        )

        operations = []

        for layer in range(self.layers):
            base = layer * 3 * self.qubits

            for q in range(self.qubits):
                operations.append(
                    (
                        "rx",
                        float(values[layer, 0, q]),
                        (q,),
                        base + q,
                    )
                )

            for q in range(self.qubits):
                operations.append(
                    (
                        "rzz",
                        float(values[layer, 1, q]),
                        (q, (q + 1) % self.qubits),
                        base + self.qubits + q,
                    )
                )

            for q in range(self.qubits):
                operations.append(
                    (
                        "ry",
                        float(values[layer, 2, q]),
                        (q,),
                        base + 2 * self.qubits + q,
                    )
                )

        return operations

    def circuit(self, parameters):
        circuit = QuantumCircuit(self.qubits)

        for name, angle, qubits, _ in self.operations(parameters):
            circuit.append(
                quantum_gate(name, angle),
                list(qubits),
            )

        return circuit

    def forward(self, parameters, packed):
        state = Statevector(packed).evolve(
            self.circuit(parameters)
        )

        return np.asarray(
            state.data, dtype=np.complex128
        )

    def generator(self, vector, name, qubits):
        matrix = np.asarray(vector).reshape(
            -1, self.dimension
        )

        q = qubits[0]

        if name == "rx":
            return matrix[
                :, self.indices ^ (1 << q)
            ].reshape(-1)

        if name == "ry":
            bit = (
                (self.indices >> q) & 1
            ).astype(np.float64)

            signs = 1j * (2.0 * bit - 1.0)

            return (
                matrix[:, self.indices ^ (1 << q)]
                * signs[None, :]
            ).reshape(-1)

        if name == "rzz":
            r = qubits[1]

            parity = (
                ((self.indices >> q) & 1)
                ^ ((self.indices >> r) & 1)
            )

            return (
                matrix
                * (1.0 - 2.0 * parity)[None, :]
            ).reshape(-1)

        raise RuntimeError("Nepoznat generator.")

    def backward(
        self,
        parameters,
        final_state,
        final_adjoint,
    ):
        state = Statevector(final_state)
        adjoint = Statevector(final_adjoint)

        gradient = np.zeros(
            self.size, dtype=np.float64
        )

        for name, angle, qubits, index in reversed(
            self.operations(parameters)
        ):
            generated = self.generator(
                state.data, name, qubits
            )

            gradient[index] = float(
                np.real(
                    -1j
                    * np.vdot(
                        adjoint.data, generated
                    )
                )
            )

            inverse = quantum_gate(name, -angle)

            state = state.evolve(
                inverse, qargs=list(qubits)
            )

            adjoint = adjoint.evolve(
                inverse, qargs=list(qubits)
            )

        finite(gradient, "Gradijent kvantnog bloka")

        return gradient, np.asarray(adjoint.data)


def build_memories(keys, values):
    """
    Za upit t:
        poznati parovi su
        (keys[0], values[1]), ...,
        (keys[t-1], values[t]).

    values[t+1] nikada ne ulazi u memoriju za upit t.

    Centriranje uklanja samostalne srednje vrednosti.
    Memorija zadrzava povezanost ulaza i ishoda.
    """

    keys = np.asarray(keys, dtype=np.complex128)
    values = np.asarray(values, dtype=np.complex128)

    require(
        keys.ndim == 2
        and values.shape == keys.shape,
        "Neispravni kontekstualni kljucevi ili vrednosti.",
    )

    finite(keys, "Kontekstualni kljucevi")
    finite(values, "Kontekstualne vrednosti")

    rows, dimension = keys.shape

    memories = np.zeros(
        (rows, dimension, dimension),
        dtype=np.complex128,
    )

    counts = np.arange(rows, dtype=np.int64)

    sum_keys = np.zeros(
        dimension, dtype=np.complex128
    )

    sum_values = np.zeros_like(sum_keys)

    sum_cross = np.zeros(
        (dimension, dimension),
        dtype=np.complex128,
    )

    for t in range(1, rows):
        key = keys[t - 1]
        value = values[t]

        sum_keys += key
        sum_values += value

        sum_cross += np.outer(
            value, np.conj(key)
        )

        mean_key = sum_keys / t
        mean_value = sum_values / t

        covariance = (
            sum_cross / t
            - np.outer(
                mean_value, np.conj(mean_key)
            )
        )

        norm = float(
            np.linalg.norm(covariance, ord="fro")
        )

        if norm > MEMORY_EPSILON:
            memories[t] = covariance / norm

    require(
        np.array_equal(
            counts, np.arange(rows)
        ),
        "Neispravni brojevi poznatih prelaza.",
    )

    finite(memories, "Kontekstualne memorije")

    return memories, counts


class ContextModel:
    def __init__(
        self,
        numbers=NUMBERS,
        picks=PICKS,
        qubits=QUBITS,
        layers=LAYERS_PER_BLOCK,
    ):
        self.numbers = int(numbers)
        self.picks = int(picks)

        self.block = QuantumBlock(
            qubits=qubits,
            layers=layers,
        )

        self.dimension = self.block.dimension
        self.size = 2 * self.block.size

        require(
            self.numbers <= self.dimension,
            "Registar ne moze predstaviti sve brojeve.",
        )

    def initial_parameters(self):
        positions = (
            np.arange(self.size, dtype=np.float64)
            + 1.0
        )

        # Deterministicki pocetak.
        # Nema poziva za slucajno generisanje.
        return (
            0.08 * np.sin(
                positions * math.sqrt(2.0) + 0.39
            )
            + 0.03 * np.cos(
                positions * math.sqrt(3.0) + 0.039
            )
        )

    def split(self, parameters):
        parameters = np.asarray(
            parameters, dtype=np.float64
        )

        require(
            parameters.shape == (self.size,),
            "Neispravan broj parametara.",
        )

        finite(parameters, "Parametri modela")

        return (
            parameters[:self.block.size],
            parameters[self.block.size:],
        )

    def forward(
        self,
        parameters,
        batch,
        memories,
        strength,
    ):
        require(
            0.0 <= strength < 0.5,
            "Jacina konteksta mora biti manja od 0.5.",
        )

        require(
            batch.dimension == self.dimension
            and memories.shape
            == (
                batch.samples,
                self.numbers,
                self.numbers,
            ),
            "Neispravan oblik konteksta.",
        )

        query_parameters, decoder_parameters = self.split(
            parameters
        )

        n = batch.samples
        m = self.numbers

        query_state = self.block.forward(
            query_parameters,
            batch.packed,
        )

        query_matrix = query_state.reshape(
            batch.padded, self.dimension
        )

        query = (
            query_matrix[:n] * math.sqrt(n)
        )

        alpha = 1.0 - strength

        filtered = alpha * query

        filtered[:, :m] += strength * np.einsum(
            "nij,nj->ni",
            memories,
            query[:, :m],
            optimize=True,
        )

        success = np.sum(
            np.abs(filtered) ** 2, axis=1
        )

        finite(success, "Verovatnoca memorijskog filtera")

        lower_bound = (1.0 - 2.0 * strength) ** 2

        require(
            np.min(success) >= lower_bound - 2.0e-8
            and np.max(success) <= 1.0 + 2.0e-8,
            "Memorijski filter nije kontrakcija.",
        )

        require(
            np.all(success > 0.0),
            "Memorijski filter ima nultu granu.",
        )

        conditional = (
            filtered / np.sqrt(success)[:, None]
        )

        decoder_input = np.zeros(
            (batch.padded, self.dimension),
            dtype=np.complex128,
        )

        decoder_input[:n] = (
            conditional / math.sqrt(n)
        )

        decoder_state = self.block.forward(
            decoder_parameters,
            decoder_input.reshape(-1),
        )

        output_matrix = decoder_state.reshape(
            batch.padded, self.dimension
        )

        output_amplitudes = (
            output_matrix[:n] * math.sqrt(n)
        )

        require(
            np.max(
                np.abs(
                    np.sum(
                        np.abs(output_amplitudes) ** 2,
                        axis=1,
                    ) - 1.0
                )
            ) < 2.0e-9,
            "Qiskit izlaz nije normalizovan.",
        )

        probabilities = (
            np.abs(
                output_amplitudes[:, :m]
            ) ** 2
        )

        finite(probabilities, "Bornove tezine")

        require(
            np.all(probabilities > 0.0),
            "Kandidat ima nultu Bornovu tezinu.",
        )

        information = (
            query_state,
            decoder_state,
            conditional,
            success,
        )

        return probabilities, information


class ContextObjective:
    def __init__(
        self,
        model,
        distribution,
        inputs,
        memories,
        targets,
        strength,
        ridge=RIDGE,
    ):
        self.model = model
        self.distribution = distribution

        self.batch = QuantumBatch(inputs)
        self.memories = memories

        self.targets = np.asarray(
            targets, dtype=np.int64
        )

        self.strength = float(strength)
        self.ridge = float(ridge)

        require(
            model.numbers == distribution.numbers
            and model.picks == distribution.picks
            and self.targets.shape
            == (
                self.batch.samples,
                distribution.picks,
            )
            and np.all(self.targets >= 0)
            and np.all(
                self.targets < distribution.numbers
            )
            and np.all(
                np.diff(self.targets, axis=1) > 0
            ),
            "Neispravna konfiguracija ciljne raspodele.",
        )

    def evaluate(self, parameters):
        probabilities, information = self.model.forward(
            parameters,
            self.batch,
            self.memories,
            self.strength,
        )

        (
            query_state,
            decoder_state,
            conditional,
            success,
        ) = information

        log_weights = np.log(probabilities)

        log_z, marginal = self.distribution.statistics(
            log_weights
        )

        target_log_weight = np.take_along_axis(
            log_weights, self.targets, axis=1
        ).sum(axis=1)

        log_probability = target_log_weight - log_z

        finite(log_probability, "Ciljna verovatnoca")

        require(
            np.max(log_probability) <= 2.0e-8,
            "Ciljna verovatnoca prelazi jedan.",
        )

        nll = -float(np.mean(log_probability))

        residual = marginal.copy()

        n = self.batch.samples
        m = self.model.numbers
        dimension = self.model.dimension

        residual[
            np.arange(n)[:, None],
            self.targets,
        ] -= 1.0

        decoder_matrix = decoder_state.reshape(
            self.batch.padded, dimension
        )

        output_adjoint = np.zeros(
            (self.batch.padded, dimension),
            dtype=np.complex128,
        )

        output_adjoint[:n, :m] = (
            residual
            * decoder_matrix[:n, :m]
            / probabilities
        )

        query_parameters, decoder_parameters = (
            self.model.split(parameters)
        )

        decoder_gradient, conditional_adjoint = (
            self.model.block.backward(
                decoder_parameters,
                decoder_state,
                output_adjoint.reshape(-1),
            )
        )

        conditional_adjoint = (
            conditional_adjoint.reshape(
                self.batch.padded, dimension
            )[:n]
        )

        projection = np.real(
            np.sum(
                np.conj(conditional)
                * conditional_adjoint,
                axis=1,
            )
        )

        filtered_adjoint = (
            conditional_adjoint
            - conditional * projection[:, None]
        ) / np.sqrt(success)[:, None]

        alpha = 1.0 - self.strength

        query_adjoint = (
            alpha * filtered_adjoint
        )

        # Adjungovano dejstvo memorije bez kopiranja
        # celog kompleksno konjugovanog kontekstualnog niza.
        memory_adjoint = np.conj(
            np.einsum(
                "nij,ni->nj",
                self.memories,
                np.conj(filtered_adjoint[:, :m]),
                optimize=True,
            )
        )

        query_adjoint[:, :m] += (
            self.strength * memory_adjoint
        )

        packed_query_adjoint = np.zeros(
            (self.batch.padded, dimension),
            dtype=np.complex128,
        )

        packed_query_adjoint[:n] = query_adjoint

        query_gradient, _ = self.model.block.backward(
            query_parameters,
            query_state,
            packed_query_adjoint.reshape(-1),
        )

        gradient = np.concatenate(
            (query_gradient, decoder_gradient)
        )

        regularization = (
            0.5
            * self.ridge
            * float(np.mean(parameters ** 2))
        )

        gradient += (
            self.ridge
            * parameters
            / self.model.size
        )

        loss = nll + regularization

        require(
            math.isfinite(loss),
            "Funkcija cilja nije konacna.",
        )

        finite(gradient, "Gradijent kontekstualnog modela")

        return loss, gradient


def memory_unitary(memory, strength, dimension):
    """
    Unitarno prosirenje kontrakcije C:

        C = (1-strength) I + strength M

        U = [ C                 sqrt(I-C C*) ]
            [ sqrt(I-C* C)      -C*         ]

    Pomocni kubit je najznacajniji bit.
    Njegova grana 0 daje C |query>.
    """

    memory = np.asarray(
        memory, dtype=np.complex128
    )

    require(
        memory.ndim == 2
        and memory.shape[0] == memory.shape[1]
        and memory.shape[0] <= dimension,
        "Neispravna memorijska matrica.",
    )

    require(
        0.0 <= strength < 0.5,
        "Neispravna jacina memorije.",
    )

    contraction = (
        (1.0 - strength)
        * np.eye(dimension, dtype=np.complex128)
    )

    m = len(memory)

    contraction[:m, :m] += (
        strength * memory
    )

    left_vectors, singular_values, right_adjoint = (
        np.linalg.svd(
            contraction,
            full_matrices=False,
        )
    )

    require(
        np.max(singular_values) <= 1.0 + 2.0e-10,
        "Operator konteksta nije kontrakcija.",
    )

    roots = np.sqrt(
        np.maximum(
            0.0,
            1.0 - singular_values * singular_values,
        )
    )

    right_vectors = right_adjoint.conj().T

    left_root = (
        left_vectors * roots[None, :]
    ) @ left_vectors.conj().T

    right_root = (
        right_vectors * roots[None, :]
    ) @ right_vectors.conj().T

    unitary = np.block(
        [
            [contraction, left_root],
            [right_root, -contraction.conj().T],
        ]
    )

    identity_error = float(
        np.max(
            np.abs(
                unitary.conj().T @ unitary
                - np.eye(2 * dimension)
            )
        )
    )

    require(
        identity_error < 3.0e-9,
        "Unitarno prosirenje memorije nije ispravno.",
    )

    return unitary, identity_error


def physical_context_check(
    model,
    parameters,
    input_state,
    memory,
    strength,
):
    input_state = np.asarray(
        input_state, dtype=np.complex128
    )

    batch = QuantumBatch(
        input_state[None, :]
    )

    probabilities, information = model.forward(
        parameters,
        batch,
        memory[None, :, :],
        strength,
    )

    expected_state = (
        information[1][:model.dimension]
    )

    expected_success = float(
        information[3][0]
    )

    unitary, unitary_error = memory_unitary(
        memory,
        strength,
        model.dimension,
    )

    query_parameters, decoder_parameters = model.split(
        parameters
    )

    circuit = QuantumCircuit(
        model.block.qubits + 1
    )

    data_qubits = list(
        range(model.block.qubits)
    )

    circuit.compose(
        model.block.circuit(query_parameters),
        qubits=data_qubits,
        inplace=True,
    )

    circuit.unitary(
        unitary,
        list(range(model.block.qubits + 1)),
        label="context",
    )

    circuit.compose(
        model.block.circuit(decoder_parameters),
        qubits=data_qubits,
        inplace=True,
    )

    initial = np.zeros(
        2 * model.dimension,
        dtype=np.complex128,
    )

    initial[:model.dimension] = input_state

    actual = Statevector(initial).evolve(circuit)

    branch = actual.data[:model.dimension]

    actual_success = float(
        np.sum(np.abs(branch) ** 2)
    )

    require(
        actual_success > 0.0,
        "Eksplicitna memorijska grana je nulta.",
    )

    normalized_branch = (
        branch / math.sqrt(actual_success)
    )

    amplitude_error = float(
        np.max(
            np.abs(
                normalized_branch - expected_state
            )
        )
    )

    probability_error = float(
        np.max(
            np.abs(
                np.abs(
                    normalized_branch[:model.numbers]
                ) ** 2
                - probabilities[0]
            )
        )
    )

    success_error = abs(
        actual_success - expected_success
    )

    require(
        amplitude_error < 2.0e-9
        and probability_error < 2.0e-9
        and success_error < 2.0e-9,
        "Eksplicitna Qiskit provera konteksta nije prosla.",
    )

    return {
        "unitary_error": unitary_error,
        "amplitude_error": amplitude_error,
        "probability_error": probability_error,
        "success_error": success_error,
        "success": actual_success,
    }


@dataclass
class Fit:
    parameters: np.ndarray
    gradient: float
    loss: float
    message: str


class EvaluationBudgetReached(Exception):
    pass


def projected_gradient(parameters, gradient):
    result = np.asarray(gradient).copy()

    lower = parameters <= -ANGLE_BOUND + 1.0e-10
    upper = parameters >= ANGLE_BOUND - 1.0e-10

    result[lower & (gradient > 0.0)] = 0.0
    result[upper & (gradient < 0.0)] = 0.0

    return float(
        np.max(np.abs(result))
    )


def fit_model(
    model,
    inputs,
    memories,
    targets,
    strength,
    label,
    warm=None,
):
    objective = ContextObjective(
        model,
        DISTRIBUTION,
        inputs,
        memories,
        targets,
        strength,
    )

    initial = (
        model.initial_parameters()
        if warm is None
        else warm.parameters.copy()
    )

    key = digest_items(
        VERSION,
        CODE_HASH,
        39,
        label,
        model.block.qubits,
        model.block.layers,
        strength,
        RIDGE,
        ANGLE_BOUND,
        MAX_ITERATIONS,
        MAX_EVALUATIONS,
        MEMORY_EPSILON,
        np.__version__,
        scipy.__version__,
        qiskit.__version__,
        inputs,
        memories,
        targets,
        initial,
    )

    path = CACHE_DIR / (
        "fit_" + key + ".npz"
    )

    parameters = initial.copy()
    iterations = 0
    done = False
    message = "Pocetak."

    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                require(
                    str(saved["version"].item()) == VERSION
                    and int(saved["seed"].item()) == 39
                    and str(saved["code_hash"].item()) == CODE_HASH,
                    "Neodgovarajuca kontrolna tacka.",
                )

                parameters = np.asarray(
                    saved["parameters"],
                    dtype=np.float64,
                ).copy()

                iterations = int(
                    saved["iterations"].item()
                )

                done = bool(saved["done"].item())
                message = str(saved["message"].item())

            require(
                parameters.shape == (model.size,)
                and 0 <= iterations <= MAX_ITERATIONS,
                "Neispravni sacuvani parametri.",
            )

            finite(parameters, "Sacuvani parametri")

            require(
                np.max(np.abs(parameters))
                <= ANGLE_BOUND + 1.0e-10,
                "Sacuvani uglovi su izvan granica.",
            )

            say(label, "| ucitana kontrolna tacka.")

        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            EOFError,
            BadZipFile,
        ) as error:
            say(
                label,
                "| kontrolna tacka nije prihvacena:",
                str(error),
            )

            parameters = initial.copy()
            iterations = 0
            done = False
            message = "Ponovljen pocetak."

    loss, gradient = objective.evaluate(parameters)

    best_parameters = parameters.copy()
    best_loss = float(loss)
    best_gradient = gradient.copy()

    calls = 0
    start = time.perf_counter()

    say()
    say(label)

    say(
        f"Uzorci: {len(inputs)} "
        f"| jacina konteksta: {strength:.2f} "
        f"| parametri: {model.size}"
    )

    def save(done_flag, status):
        atomic_npz(
            path,
            version=np.array(VERSION),
            seed=np.array(39, dtype=np.int64),
            code_hash=np.array(CODE_HASH),
            parameters=best_parameters,
            iterations=np.array(
                min(iterations, MAX_ITERATIONS),
                dtype=np.int64,
            ),
            done=np.array(bool(done_flag)),
            message=np.array(str(status)),
        )

    def function(candidate):
        nonlocal calls
        nonlocal best_parameters, best_loss, best_gradient

        if calls >= MAX_EVALUATIONS:
            raise EvaluationBudgetReached()

        calls += 1

        value, derivative = objective.evaluate(
            candidate
        )

        if value < best_loss:
            best_loss = float(value)
            best_parameters = np.array(
                candidate, copy=True
            )
            best_gradient = np.array(
                derivative, copy=True
            )

        if calls == 1 or calls % 5 == 0:
            say(
                f"Evaluacija {calls}/{MAX_EVALUATIONS} "
                f"| cilj {best_loss:.9f} "
                f"| gradijent "
                f"{projected_gradient(best_parameters, best_gradient):.3e} "
                f"| proteklo "
                f"{time.perf_counter() - start:.1f} s"
            )

        return value, derivative

    def callback(_candidate):
        nonlocal iterations

        iterations += 1

        if iterations % 3 == 0:
            save(False, "Trening u toku.")

    try:
        if not done and iterations < MAX_ITERATIONS:
            optimization = minimize(
                function,
                best_parameters,
                method="L-BFGS-B",
                jac=True,
                bounds=[
                    (-ANGLE_BOUND, ANGLE_BOUND)
                ] * model.size,
                callback=callback,
                options={
                    "maxiter": MAX_ITERATIONS - iterations,
                    "maxfun": MAX_EVALUATIONS,
                    "maxls": 20,
                    "maxcor": 10,
                    "ftol": 1.0e-11,
                    "gtol": 1.0e-6,
                },
            )

            message = str(
                optimization.message
            )

        elif not done:
            message = "Dostignut budzet iteracija."

    except EvaluationBudgetReached:
        message = "Dostignut budzet evaluacija."

    except KeyboardInterrupt:
        save(False, "Korisnicki prekid.")
        raise

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        message = (
            "Zadrzan najbolji numericki vazeci kandidat. "
            + str(error)
        )

    checked_loss, checked_gradient = objective.evaluate(
        best_parameters
    )

    require(
        abs(checked_loss - best_loss)
        < 2.0e-8 * (1.0 + abs(best_loss)),
        "Ponovljeni proracun cilja se ne slaze.",
    )

    gradient_norm = projected_gradient(
        best_parameters,
        checked_gradient,
    )

    save(True, message)

    say(f"Najbolji cilj: {checked_loss:.9f}")
    say(f"Max|projektovani gradijent|: {gradient_norm:.3e}")
    say("Optimizator:", message)

    if gradient_norm > GRADIENT_LIMIT:
        say(
            "DIJAGNOSTIKA: stacionarnost nije potvrdjena; "
            "koristi se najbolji vazeci kandidat."
        )

    return Fit(
        parameters=best_parameters.copy(),
        gradient=gradient_norm,
        loss=float(checked_loss),
        message=message,
    )


def predict(
    model,
    fit,
    inputs,
    memories,
    strength,
):
    probabilities, information = model.forward(
        fit.parameters,
        QuantumBatch(inputs),
        memories,
        strength,
    )

    return (
        np.log(probabilities),
        information[3],
    )


def evaluate(
    model,
    fit,
    inputs,
    memories,
    targets,
    strength,
    with_hits=False,
):
    log_weights, _ = predict(
        model,
        fit,
        inputs,
        memories,
        strength,
    )

    nll = -float(
        np.mean(
            DISTRIBUTION.log_probabilities(
                log_weights, targets
            )
        )
    )

    if not with_hits:
        return nll

    choices = np.argsort(
        -log_weights,
        axis=1,
        kind="stable",
    )[:, :PICKS]

    occupancy = np.zeros(
        (len(targets), NUMBERS),
        dtype=np.int8,
    )

    occupancy[
        np.arange(len(targets))[:, None],
        targets,
    ] = 1

    hits = np.take_along_axis(
        occupancy,
        choices,
        axis=1,
    ).sum(axis=1)

    histogram = np.bincount(
        hits.astype(np.int64),
        minlength=PICKS + 1,
    )

    return nll, histogram


def encode_history(draws):
    occupancy = np.zeros(
        (len(draws), NUMBERS),
        dtype=np.float64,
    )

    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1.0

    rows = np.arange(len(draws))

    current = occupancy[rows]
    previous = occupancy[np.maximum(rows - 1, 0)]
    older = occupancy[np.maximum(rows - 2, 0)]

    strength = (
        1.0 + current
        + 0.5 * previous
        + 0.25 * older
    )

    phase = math.pi * (
        0.37 * current
        - 0.23 * previous
        + 0.11 * older
    )

    inputs = np.zeros(
        (len(draws), DIMENSION),
        dtype=np.complex128,
    )

    inputs[:, :NUMBERS] = (
        np.sqrt(strength) * np.exp(1j * phase)
    )

    inputs /= np.sqrt(
        np.sum(np.abs(inputs) ** 2, axis=1)
    )[:, None]

    return inputs


def encode_values(draws):
    """
    Amplitudno kodiranje stvarnih celih kombinacija.
    Ne racuna istorijske ucestalosti.
    """

    positions = (
        np.arange(1, NUMBERS + 1, dtype=np.float64)
        / NUMBERS
    )

    phases = math.pi * (
        positions + 0.3 * positions ** 2
    )

    values = np.zeros(
        (len(draws), NUMBERS),
        dtype=np.complex128,
    )

    values[
        np.arange(len(draws))[:, None],
        draws,
    ] = (
        np.exp(1j * phases[draws])
        / math.sqrt(PICKS)
    )

    return values


def prepare_history(draws):
    inputs = encode_history(draws)
    values = encode_values(draws)

    memories, counts = build_memories(
        inputs[:, :NUMBERS],
        values,
    )

    require(
        counts[-1] == len(draws) - 1,
        "NEXT memorija ne sadrzi sve istorijske prelaze.",
    )

    return inputs, memories, counts


def load_draws(path):
    raw = path.read_bytes()

    fingerprint = hashlib.sha256(
        raw
    ).hexdigest()

    rows = []
    blanks = 0

    for line_number, row in enumerate(
        csv.reader(
            raw.decode("utf-8-sig").splitlines()
        ),
        start=1,
    ):
        if not row or all(
            not value.strip() for value in row
        ):
            blanks += 1
            continue

        require(
            len(row) == PICKS,
            f"{path.name}, red {line_number}: "
            "potrebno je sedam kolona.",
        )

        try:
            values = [
                int(value.strip()) for value in row
            ]

        except ValueError as error:
            raise RuntimeError(
                f"{path.name}, red {line_number}: "
                "vrednosti moraju biti celi brojevi."
            ) from error

        require(
            len(set(values)) == PICKS
            and min(values) >= 1
            and max(values) <= NUMBERS,
            f"{path.name}, red {line_number}: "
            "neispravna kombinacija.",
        )

        rows.append(
            sorted(value - 1 for value in values)
        )

    require(
        len(rows) >= 100,
        f"{path.name}: potrebno je najmanje 100 kola.",
    )

    draws = np.asarray(
        rows, dtype=np.int64
    )

    say()
    say("ULAZ:", path.name)
    say(f"Redova: {len(draws)} | praznih redova: {blanks}")
    say("SHA256:", fingerprint)
    say("Redosled kola ostaje redosled iz CSV-a.")

    return draws, fingerprint


def self_checks():
    say()
    say("Pokrecem matematicke i Qiskit samoprovere.")

    distribution = FixedSizeDistribution(6, 3)

    x = np.array(
        [[-0.2, 0.1, -0.7, 0.4, -0.1, 0.2]],
        dtype=np.float64,
    )

    subsets = np.array(
        list(combinations(range(6), 3)),
        dtype=np.int64,
    )

    weights = np.exp(
        x[0, subsets].sum(axis=1)
    )

    probabilities = weights / weights.sum()

    log_z, marginal = distribution.statistics(x)

    exact_marginal = np.zeros(6)

    for subset, probability in zip(
        subsets, probabilities
    ):
        exact_marginal[subset] += probability

    require(
        abs(
            log_z[0] - math.log(float(weights.sum()))
        ) < 2.0e-12,
        "Normalizacija ne prolazi nabrajanje.",
    )

    require(
        np.max(
            np.abs(
                marginal[0] - exact_marginal
            )
        ) < 2.0e-12,
        "Marginalne verovatnoce nisu ispravne.",
    )

    selected, log_probability = (
        distribution.maximum(x[0])
    )

    require(
        np.array_equal(
            selected,
            subsets[int(np.argmax(weights))],
        ),
        "Globalni maksimum nije ispravan.",
    )

    require(
        abs(
            math.exp(log_probability)
            - float(np.max(probabilities))
        ) < 2.0e-12,
        "Verovatnoca maksimuma nije ispravna.",
    )

    require(
        abs(
            distribution.independent_log_z(x[0])
            - log_z[0]
        ) < 2.0e-12,
        "Nezavisna normalizacija nije ispravna.",
    )

    for picks in (1, 6):
        boundary = FixedSizeDistribution(6, picks)

        value = boundary.statistics(
            np.zeros((1, 6))
        )[0][0]

        require(
            abs(
                value - math.log(math.comb(6, picks))
            ) < 2.0e-12,
            "Granicni slucaj nije ispravan.",
        )

    keys = np.empty(
        (6, 8), dtype=np.complex128
    )

    values = np.empty(
        (6, 6), dtype=np.complex128
    )

    for row in range(6):
        for j in range(8):
            keys[row, j] = (
                1.0
                + 0.15 * math.sin((row + 1) * (j + 1))
            ) * np.exp(0.19j * row * j)

        for j in range(6):
            values[row, j] = (
                0.7
                + 0.2 * math.cos((row + 2) * (j + 1))
            ) * np.exp(0.23j * (row + 1) * j)

    keys /= np.sqrt(
        np.sum(np.abs(keys) ** 2, axis=1)
    )[:, None]

    values /= np.sqrt(
        np.sum(np.abs(values) ** 2, axis=1)
    )[:, None]

    memories, counts = build_memories(
        keys[:, :6], values
    )

    prefix_memories, prefix_counts = build_memories(
        keys[:4, :6], values[:4]
    )

    require(
        np.array_equal(
            counts[:4], prefix_counts
        )
        and np.max(
            np.abs(
                memories[:4] - prefix_memories
            )
        ) < 2.0e-12,
        "Kontekst nije kauzalan.",
    )

    direct_cross = sum(
        (
            np.outer(
                values[t + 1],
                np.conj(keys[t, :6]),
            )
            for t in range(3)
        ),
        np.zeros((6, 6), dtype=np.complex128),
    ) / 3.0

    direct_covariance = (
        direct_cross
        - np.outer(
            values[1:4].mean(axis=0),
            np.conj(
                keys[:3, :6].mean(axis=0)
            ),
        )
    )

    direct_covariance /= np.linalg.norm(
        direct_covariance,
        ord="fro",
    )

    require(
        np.max(
            np.abs(
                direct_covariance - memories[3]
            )
        ) < 2.0e-12,
        "Memorija ne cuva ispravne parove.",
    )

    require(
        np.max(np.abs(memories[0])) == 0.0
        and np.max(np.abs(memories[1])) == 0.0,
        "Prazna ili jednoprimerna centrirana memorija nije nulta.",
    )

    model = ContextModel(
        numbers=6,
        picks=3,
        qubits=3,
        layers=1,
    )

    inputs = keys[2:5]
    context = memories[2:5]

    targets = np.array(
        [[0, 2, 5], [1, 3, 4], [0, 1, 4]],
        dtype=np.int64,
    )

    parameters = model.initial_parameters()

    require(
        np.array_equal(
            parameters,
            model.initial_parameters(),
        ),
        "Inicijalizacija nije deterministicka.",
    )

    objective = ContextObjective(
        model,
        distribution,
        inputs,
        context,
        targets,
        strength=0.40,
    )

    loss, gradient = objective.evaluate(
        parameters
    )

    require(
        math.isfinite(loss),
        "Samoprovera cilja nije konacna.",
    )

    epsilon = 2.0e-6
    gradient_error = 0.0

    for index in range(model.size):
        plus = parameters.copy()
        minus = parameters.copy()

        plus[index] += epsilon
        minus[index] -= epsilon

        numerical = (
            objective.evaluate(plus)[0]
            - objective.evaluate(minus)[0]
        ) / (2.0 * epsilon)

        error = abs(
            numerical - gradient[index]
        )

        gradient_error = max(
            gradient_error, error
        )

        require(
            error
            < 3.0e-5 * (
                1.0 + abs(gradient[index])
            ),
            "Gradijent kontekstualnog modela nije ispravan.",
        )

    quantum_check = physical_context_check(
        model,
        parameters,
        inputs[0],
        context[0],
        strength=0.40,
    )

    batch = QuantumBatch(inputs)

    q_context, _ = model.forward(
        parameters,
        batch,
        context,
        0.40,
    )

    q_without, _ = model.forward(
        parameters,
        batch,
        np.zeros_like(context),
        0.40,
    )

    context_marginal = distribution.statistics(
        np.log(q_context)
    )[1]

    without_marginal = distribution.statistics(
        np.log(q_without)
    )[1]

    require(
        np.max(
            np.abs(
                context_marginal - without_marginal
            )
        ) > 1.0e-8,
        "Kontekst ne menja izlaz na matematickom primeru.",
    )

    say("Samoprovere su prosle.")
    say(f"Greska gradijenta: {gradient_error:.3e}")

    say(
        f"Greska unitarnosti memorije: "
        f"{quantum_check['unitary_error']:.3e}"
    )

    say(
        f"Greska eksplicitnih Qiskit amplituda: "
        f"{quantum_check['amplitude_error']:.3e}"
    )

    say(
        f"Greska verovatnoce pomocne grane: "
        f"{quantum_check['success_error']:.3e}"
    )


def resource_report(loaded, model):
    largest = max(
        len(draws)
        for _, draws, _ in loaded
    )

    padded = next_power_of_two(largest)

    memory_bytes = (
        largest * NUMBERS * NUMBERS * 16
    )

    state_bytes = (
        padded * DIMENSION * 16
    )

    dp_bytes = (
        largest
        * (NUMBERS + 1)
        * (PICKS + 1)
        * 8
    )

    estimate = (
        384 * 1024**2
        + memory_bytes
        + 14 * state_bytes
        + 8 * dp_bytes
        + 8 * largest * DIMENSION * 16
    )

    say()
    say("PROCENA RESURSA")

    say(
        f"Kontekstualni operator: "
        f"{NUMBERS} x {NUMBERS}"
    )

    say(
        f"Naucenih kvantnih uglova: {model.size}"
    )

    say(
        f"Memorija istorijskih operatora: "
        f"{memory_bytes / 1024**2:.1f} MiB"
    )

    say(
        f"Konzervativna procena radne memorije: "
        f"{estimate / 1024**3:.2f} GiB"
    )

    say(
        "Procena nije izmereni vrh memorije procesa."
    )

    require(
        estimate <= 3.0 * 1024**3,
        "Procenjena memorija prelazi tri GiB. "
        "Trening nije zapocet.",
    )


def process_file(
    path,
    draws,
    fingerprint,
    model,
):
    say()
    say("=" * 70)
    say("PRIPREMA KONTEKSTA:", path.name)

    all_inputs, all_memories, counts = (
        prepare_history(draws)
    )

    inputs = all_inputs[:-1]
    memories = all_memories[:-1]
    targets = draws[1:]

    count = len(targets)

    cut60 = int(0.60 * count)
    cut70 = int(0.70 * count)
    cut80 = int(0.80 * count)

    folds = (
        (cut60, cut70),
        (cut70, cut80),
    )

    candidates = []

    say(f"Istorijskih prelaza: {count}")

    say(
        f"Poznatih prelaza u NEXT kontekstu: "
        f"{int(counts[-1])}"
    )

    for strength in CONTEXT_STRENGTHS:
        warm = None
        weighted_nll = 0.0
        validation_rows = 0

        for fold_index, (train_end, valid_end) in enumerate(folds):
            fit = fit_model(
                model,
                inputs[:train_end],
                memories[:train_end],
                targets[:train_end],
                strength,
                (
                    f"{path.name} "
                    f"| kontekst {strength:.2f} "
                    f"| validacija {fold_index + 1}"
                ),
                warm=warm,
            )

            score = evaluate(
                model,
                fit,
                inputs[train_end:valid_end],
                memories[train_end:valid_end],
                targets[train_end:valid_end],
                strength,
            )

            rows = valid_end - train_end

            weighted_nll += score * rows
            validation_rows += rows
            warm = fit

            say(f"Validacioni NLL: {score:.9f}")

        candidates.append(
            {
                "strength": strength,
                "validation": (
                    weighted_nll / validation_rows
                ),
                "warm": warm,
            }
        )

    best_validation = min(
        candidate["validation"]
        for candidate in candidates
    )

    equivalent = [
        candidate
        for candidate in candidates
        if candidate["validation"]
        <= best_validation + VALIDATION_TIE
    ]

    selected = min(
        equivalent,
        key=lambda candidate: candidate["strength"],
    )

    strength = selected["strength"]

    say()
    say("IZBOR JACINE KONTEKSTA")

    for candidate in candidates:
        say(
            f"Jacina {candidate['strength']:.2f} "
            f"| validacioni NLL "
            f"{candidate['validation']:.9f}"
        )

    say(
        f"Izabrana jacina konteksta: {strength:.2f}"
    )

    say(
        f"Pri razlici do {VALIDATION_TIE:.6f} NLL "
        "prednost ima slabiji filter."
    )

    test_fit = fit_model(
        model,
        inputs[:cut80],
        memories[:cut80],
        targets[:cut80],
        strength,
        f"{path.name} | odvojeni test",
        warm=selected["warm"],
    )

    test_nll, histogram = evaluate(
        model,
        test_fit,
        inputs[cut80:],
        memories[cut80:],
        targets[cut80:],
        strength,
        with_hits=True,
    )

    say()
    say(f"Odvojeni test NLL: {test_nll:.9f}")
    say("Test ne menja izabranu jacinu konteksta.")

    final_fit = fit_model(
        model,
        inputs,
        memories,
        targets,
        strength,
        f"{path.name} | zavrsni trening cele istorije",
        warm=test_fit,
    )

    next_input = all_inputs[-1:]
    next_memory = all_memories[-1:]

    next_log_weights, success = predict(
        model,
        final_fit,
        next_input,
        next_memory,
        strength,
    )

    next_log_weights = next_log_weights[0]

    log_z, marginal = DISTRIBUTION.statistics(
        next_log_weights[None, :]
    )

    selected_numbers, log_probability = (
        DISTRIBUTION.maximum(
            next_log_weights
        )
    )

    alternative_log_z = (
        DISTRIBUTION.independent_log_z(
            next_log_weights
        )
    )

    normalization_error = abs(
        alternative_log_z - float(log_z[0])
    )

    require(
        normalization_error < 2.0e-9,
        "Nezavisna NEXT normalizacija nije prosla.",
    )

    normalization_ratio = math.exp(
        alternative_log_z - float(log_z[0])
    )

    quantum_check = physical_context_check(
        model,
        final_fit.parameters,
        next_input[0],
        next_memory[0],
        strength,
    )

    without_log_weights, _ = predict(
        model,
        final_fit,
        next_input,
        np.zeros_like(next_memory),
        strength,
    )

    without_marginal = DISTRIBUTION.statistics(
        without_log_weights
    )[1][0]

    context_effect = float(
        np.max(
            np.abs(
                marginal[0] - without_marginal
            )
        )
    )

    recent_log_weights, _ = predict(
        model,
        final_fit,
        all_inputs[-8:],
        all_memories[-8:],
        strength,
    )

    recent_marginal = DISTRIBUTION.statistics(
        recent_log_weights
    )[1]

    history_sensitivity = float(
        np.max(
            np.ptp(recent_marginal, axis=0)
        )
    )

    numbers = tuple(
        int(value + 1)
        for value in selected_numbers
    )

    warnings = []

    if maximum_run(numbers) >= 5:
        warnings.append(
            "NEXT sadrzi najmanje pet uzastopnih brojeva."
        )

    if numbers[-1] - numbers[0] <= 9:
        warnings.append(
            "Svih sedam brojeva nalazi se u rasponu od deset brojeva."
        )

    if any(
        np.array_equal(
            selected_numbers, historical
        )
        for historical in draws
    ):
        warnings.append(
            "NEXT ponavlja istorijsku kombinaciju."
        )

    if np.array_equal(
        selected_numbers, draws[-1]
    ):
        warnings.append(
            "NEXT je jednak poslednjem istorijskom kolu."
        )

    if context_effect < 1.0e-8:
        warnings.append(
            "Kontekst veoma slabo menja NEXT raspodelu."
        )

    if history_sensitivity < 1.0e-8:
        warnings.append(
            "Izlaz slabo reaguje na proverene istorijske upite."
        )

    if final_fit.gradient > GRADIENT_LIMIT:
        warnings.append(
            "Stacionarnost zavrsnog treninga nije potvrdjena."
        )

    if test_fit.gradient > GRADIENT_LIMIT:
        warnings.append(
            "Test model ima nepotvrdjenu stacionarnost."
        )

    result_key = digest_items(
        VERSION,
        CODE_HASH,
        fingerprint,
        strength,
        final_fit.parameters,
    )

    result_path = CACHE_DIR / (
        "result_" + result_key + ".npz"
    )

    atomic_npz(
        result_path,
        version=np.array(VERSION),
        seed=np.array(39, dtype=np.int64),
        code_hash=np.array(CODE_HASH),
        data_sha256=np.array(fingerprint),
        parameters=final_fit.parameters,
        strength=np.array(strength),
        next_numbers=selected_numbers + 1,
        next_log_probability=np.array(log_probability),
        inclusion=marginal[0],
        inclusion_without_context=without_marginal,
        next_memory=next_memory[0],
        known_context_pairs=np.array(
            counts[-1], dtype=np.int64
        ),
        validation_nll=np.array(
            selected["validation"]
        ),
        test_nll=np.array(test_nll),
    )

    say()
    say("REZULTAT:", path.name)
    say("NEXT:", " ".join(map(str, numbers)))
    say(f"Jacina konteksta: {strength:.2f}")

    say(
        f"Poznatih istorijskih prelaza u kontekstu: "
        f"{int(counts[-1])}"
    )

    say(
        f"Validacioni NLL: "
        f"{selected['validation']:.9f}"
    )

    say(f"Odvojeni test NLL: {test_nll:.9f}")

    say(
        f"Modelna verovatnoca NEXT: "
        f"{math.exp(log_probability):.12g}"
    )

    say(
        f"Zbir svih verovatnoca: "
        f"{normalization_ratio:.12f}"
    )

    say(
        f"Greska logaritamske normalizacije: "
        f"{normalization_error:.3e}"
    )

    say(
        f"Zbir verovatnoca ukljucivanja: "
        f"{marginal[0].sum():.12f}"
    )

    say(
        f"Kombinacija obuhvacenih raspodelom: "
        f"{math.comb(NUMBERS, PICKS):,}"
    )

    say(
        "Maksimum je tacan za naucenu raspodelu "
        "proizvoda Bornovih tezina uz kardinalnost sedam."
    )

    say(
        f"Verovatnoca pomocne grane filtera: "
        f"{float(success[0]):.9f}"
    )

    say(
        f"Greska unitarnosti memorijskog kola: "
        f"{quantum_check['unitary_error']:.3e}"
    )

    say(
        f"Greska eksplicitnih Qiskit amplituda: "
        f"{quantum_check['amplitude_error']:.3e}"
    )

    say(
        f"Greska eksplicitnih Bornovih tezina: "
        f"{quantum_check['probability_error']:.3e}"
    )

    say(
        f"Promena raspodele uklanjanjem konteksta: "
        f"{context_effect:.9e}"
    )

    say(
        "To je dijagnostika uticaja konteksta "
        "uz iste uglove, a ne dokaz poboljsanja predikcije."
    )

    say(
        f"Osetljivost na istoriju i upit: "
        f"{history_sensitivity:.9e}"
    )

    say(
        f"Zavrsni projektovani gradijent: "
        f"{final_fit.gradient:.3e}"
    )

    say("Test histogram pogodaka:")

    for hits, frequency in enumerate(histogram):
        say(f"{hits}: {int(frequency)}")

    mean_hits = float(
        np.dot(
            np.arange(PICKS + 1),
            histogram,
        ) / histogram.sum()
    )

    say(
        f"Prosecan broj pogodaka na testu: "
        f"{mean_hits:.6f}"
    )

    say("Verovatnoce ukljucivanja u NEXT raspodeli:")

    for number, probability in enumerate(
        marginal[0], start=1
    ):
        say(f"{number:2d}: {probability:.9f}")

    say("Sacuvani rezultat i kontekst:", str(result_path))

    for warning in warnings:
        say("DIJAGNOSTIKA:", warning)

    if warnings:
        say(
            "Rezultat nije menjan radi uklanjanja "
            "dijagnostickog obrasca."
        )

    return {
        "name": path.name,
        "next": numbers,
        "strength": strength,
        "test_nll": test_nll,
        "inclusion": marginal[0].copy(),
        "context_effect": context_effect,
        "history_sensitivity": history_sensitivity,
        "memory": next_memory[0].copy(),
        "last": tuple(
            int(value + 1)
            for value in draws[-1]
        ),
    }


def main():
    say("QUANTUM TABULAR IN-CONTEXT REGRESSION V1")
    say("Python:", sys.version.split()[0])
    say("Qiskit:", qiskit.__version__)
    say("NumPy:", np.__version__)
    say("SciPy:", scipy.__version__)
    say("random.seed(39)")
    say("np.random.seed(39)")
    say("SHA256 koda:", CODE_HASH)
    say()

    say(
        "Regresija uslovne raspodele Loto kombinacije."
    )

    say(
        "Nema klasifikatora ni klasicnog "
        "prediktivnog regresora."
    )

    say(
        "Nema sinteticnog pretreninga."
    )

    say(
        "Model uci iz stvarnih hronoloskih epizoda "
        "svakog CSV-a zasebno."
    )

    say(
        "Kontekst je centrirana kompleksna memorija "
        "uparenih istorijskih prelaza."
    )

    say(
        "Svi poznati raniji prelazi ulaze u memoriju."
    )

    say(
        "Tri poslednja kola kodiraju pojedinacni upit; "
        "ne zamenjuju celu kontekstualnu istoriju."
    )

    say(
        "Qiskit izvrsava nauceno kolo upita "
        "i nauceno izlazno kolo."
    )

    say(
        "Kontekstualni filter racuna se analiticki "
        "u treningu."
    )

    say(
        "Samoprovere i NEXT proveravaju isti filter "
        "eksplicitnim Qiskit kolom sa pomocnim kubitom."
    )

    say(
        "Kvantno stanje sadrzi sest kubita podataka."
    )

    say(
        "Sedam nezavisnih identicnih priprema koristi "
        "isti nauceni pojedinacni izlaz."
    )

    say(
        "Izmedju sedam izlaznih registara "
        "nema kvantnih kapija."
    )

    say(
        "Normalizacija i optimizator su klasicni."
    )

    say(
        "Pri predikciji uglovi ostaju fiksni; "
        "promena konteksta menja amplitude."
    )

    say(
        "Validacija: 60%-70% i 70%-80% prelaza."
    )

    say(
        "Odvojeni test: poslednjih 20% prelaza."
    )

    say(
        "Validacija i test su jedan korak unapred."
    )

    say(
        "Kontekst se prosiruje samo ranije poznatim "
        "stvarnim kolima; parametri se tada ne menjaju."
    )

    say(
        "Zavrsni trening koristi sve istorijske prelaze."
    )

    say(
        f"Budzet po treningu: {MAX_ITERATIONS} iteracija "
        f"i {MAX_EVALUATIONS} optimizacionih evaluacija."
    )

    say(
        "Pocetna i zavrsna provera funkcije cilja "
        "rade se dodatno."
    )

    say(
        "Lokalna optimizacija ne garantuje "
        "globalni optimum parametara."
    )

    loaded = []

    for path in FILES:
        draws, fingerprint = load_draws(path)

        loaded.append(
            (path, draws, fingerprint)
        )

    model = ContextModel()

    resource_report(loaded, model)
    self_checks()

    CACHE_DIR.mkdir(
        parents=True, exist_ok=True
    )

    say()
    say("Kontrolne tacke:", str(CACHE_DIR))

    results = []

    for path, draws, fingerprint in loaded:
        results.append(
            process_file(
                path,
                draws,
                fingerprint,
                model,
            )
        )

    say()
    say("=" * 70)
    say("ZAVRSNI NEXT REZULTATI")

    for result in results:
        say(
            f"{result['name']} | NEXT: "
            + " ".join(map(str, result["next"]))
            + f" | kontekst: {result['strength']:.2f}"
            + f" | test NLL: {result['test_nll']:.9f}"
        )

    for first, second in combinations(results, 2):
        if first["next"] == second["next"]:
            probability_difference = float(
                np.max(
                    np.abs(
                        first["inclusion"]
                        - second["inclusion"]
                    )
                )
            )

            memory_difference = float(
                np.linalg.norm(
                    first["memory"]
                    - second["memory"],
                    ord="fro",
                )
            )

            say()
            say(
                "DIJAGNOSTIKA: isti NEXT za",
                first["name"],
                "i",
                second["name"],
            )

            say(
                f"Najveca razlika verovatnoca ukljucivanja: "
                f"{probability_difference:.9e}"
            )

            say(
                f"Razlika kontekstualnih operatora: "
                f"{memory_difference:.9e}"
            )

            say(
                f"Efekti uklanjanja konteksta: "
                f"{first['context_effect']:.9e} i "
                f"{second['context_effect']:.9e}"
            )

            say(
                "Isto poslednje istorijsko kolo:",
                first["last"] == second["last"],
            )

            say(
                "Rezultat nije menjan radi "
                "vestackog razlikovanja."
            )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        say()
        say("PREKINUTO NA ZAHTEV KORISNIKA.")

        say(
            "Tokom treninga cuva se najbolji kandidat "
            "za nastavak."
        )

        sys.exit(130)

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        say()
        say("ZAUSTAVLJENO:", str(error))

        say(
            "Zavrsni skup od tri NEXT rezultata "
            "nije potvrdjen."
        )

        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM TABULAR IN-CONTEXT REGRESSION V1
Python: 3.11.13
Qiskit: 1.4.4
NumPy: 1.26.4
SciPy: 1.17.1
random.seed(39)
np.random.seed(39)
SHA256 koda: 68a344da080d77464af954b2a6092532bd48745f1ceaf8c3e2730d5910d6b19c

Regresija uslovne raspodele Loto kombinacije.
Nema klasifikatora ni klasicnog prediktivnog regresora.
Nema sinteticnog pretreninga.
Model uci iz stvarnih hronoloskih epizoda svakog CSV-a zasebno.
Kontekst je centrirana kompleksna memorija uparenih istorijskih prelaza.
Svi poznati raniji prelazi ulaze u memoriju.
Tri poslednja kola kodiraju pojedinacni upit; ne zamenjuju celu kontekstualnu istoriju.
Qiskit izvrsava nauceno kolo upita i nauceno izlazno kolo.
Kontekstualni filter racuna se analiticki u treningu.
Samoprovere i NEXT proveravaju isti filter eksplicitnim Qiskit kolom sa pomocnim kubitom.
Kvantno stanje sadrzi sest kubita podataka.
Sedam nezavisnih identicnih priprema koristi isti nauceni pojedinacni izlaz.
Izmedju sedam izlaznih registara nema kvantnih kapija.
Normalizacija i optimizator su klasicni.
Pri predikciji uglovi ostaju fiksni; promena konteksta menja amplitude.
Validacija: 60%-70% i 70%-80% prelaza.
Odvojeni test: poslednjih 20% prelaza.
Validacija i test su jedan korak unapred.
Kontekst se prosiruje samo ranije poznatim stvarnim kolima; parametri se tada ne menjaju.
Zavrsni trening koristi sve istorijske prelaze.
Budzet po treningu: 48 iteracija i 80 optimizacionih evaluacija.
Pocetna i zavrsna provera funkcije cilja rade se dodatno.
Lokalna optimizacija ne garantuje globalni optimum parametara.

ULAZ: loto7_4712_k87.csv
Redova: 4712 | praznih redova: 0
SHA256: d5504ffa43c8d7ce477ec31e3a0fff18e06ddae6a9fda656fda8813439750808
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4712_k87_loto_2978.csv
Redova: 2978 | praznih redova: 0
SHA256: 0b1abfe53d71a4c52a467be9254d094b3e48113ae875c2a154dc6229e763d1a2
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4712_k87_loto_plus_1734.csv
Redova: 1734 | praznih redova: 0
SHA256: f2fd75c1e476476a5691c74e942cfee2e4628ee0662c68433f46a7d9efd32b9e
Redosled kola ostaje redosled iz CSV-a.

PROCENA RESURSA
Kontekstualni operator: 39 x 39
Naucenih kvantnih uglova: 36
Memorija istorijskih operatora: 109.4 MiB
Konzervativna procena radne memorije: 0.72 GiB
Procena nije izmereni vrh memorije procesa.

Pokrecem matematicke i Qiskit samoprovere.
Samoprovere su prosle.
Greska gradijenta: 4.048e-10
Greska unitarnosti memorije: 1.110e-15
Greska eksplicitnih Qiskit amplituda: 2.289e-16
Greska verovatnoce pomocne grane: 5.551e-17

Kontrolne tacke: /Quantum_Tabular_InContext_Regression/Quantum_Tabular_InContext_Regression_V1_checkpoints

======================================================================
PRIPREMA KONTEKSTA: loto7_4712_k87.csv
Istorijskih prelaza: 4711
Poznatih prelaza u NEXT kontekstu: 4711

loto7_4712_k87.csv | kontekst 0.20 | validacija 1
Uzorci: 2826 | jacina konteksta: 0.20 | parametri: 36
Evaluacija 1/80 | cilj 17.000233977 | gradijent 7.493e-01 | proteklo 0.2 s
Evaluacija 5/80 | cilj 16.758422405 | gradijent 6.358e-02 | proteklo 1.1 s
Evaluacija 10/80 | cilj 16.754210152 | gradijent 5.098e-03 | proteklo 2.1 s
Evaluacija 15/80 | cilj 16.753453578 | gradijent 6.262e-03 | proteklo 3.2 s
Evaluacija 20/80 | cilj 16.752816705 | gradijent 2.016e-02 | proteklo 4.3 s
Evaluacija 25/80 | cilj 16.749145250 | gradijent 4.488e-02 | proteklo 5.3 s
Evaluacija 30/80 | cilj 16.747735950 | gradijent 1.115e-02 | proteklo 6.4 s
Evaluacija 35/80 | cilj 16.747331509 | gradijent 1.138e-02 | proteklo 7.4 s
Evaluacija 40/80 | cilj 16.746908723 | gradijent 2.862e-03 | proteklo 8.5 s
Evaluacija 45/80 | cilj 16.746779191 | gradijent 3.015e-03 | proteklo 9.6 s
Evaluacija 50/80 | cilj 16.746397631 | gradijent 6.779e-03 | proteklo 10.6 s
Evaluacija 55/80 | cilj 16.746276807 | gradijent 5.966e-03 | proteklo 11.7 s
Najbolji cilj: 16.746224102
Max|projektovani gradijent|: 9.693e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.
Validacioni NLL: 16.756017047

loto7_4712_k87.csv | kontekst 0.20 | validacija 2
Uzorci: 3297 | jacina konteksta: 0.20 | parametri: 36
Evaluacija 1/80 | cilj 16.747670441 | gradijent 2.246e-02 | proteklo 0.3 s
Evaluacija 5/80 | cilj 16.747356150 | gradijent 4.113e-03 | proteklo 1.2 s
Evaluacija 10/80 | cilj 16.747190065 | gradijent 4.831e-03 | proteklo 2.3 s
Evaluacija 15/80 | cilj 16.747025691 | gradijent 2.489e-03 | proteklo 3.5 s
Evaluacija 20/80 | cilj 16.746913206 | gradijent 7.656e-03 | proteklo 4.6 s
Evaluacija 25/80 | cilj 16.746709496 | gradijent 2.654e-03 | proteklo 5.7 s
Evaluacija 30/80 | cilj 16.746384374 | gradijent 9.948e-03 | proteklo 6.8 s
Evaluacija 35/80 | cilj 16.746134571 | gradijent 3.591e-03 | proteklo 7.9 s
Evaluacija 40/80 | cilj 16.745988689 | gradijent 5.312e-03 | proteklo 9.0 s
Evaluacija 45/80 | cilj 16.745891630 | gradijent 2.328e-03 | proteklo 10.2 s
Evaluacija 50/80 | cilj 16.745760782 | gradijent 1.981e-03 | proteklo 11.3 s
Najbolji cilj: 16.745705988
Max|projektovani gradijent|: 5.432e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.
Validacioni NLL: 16.759222598

loto7_4712_k87.csv | kontekst 0.45 | validacija 1
Uzorci: 2826 | jacina konteksta: 0.45 | parametri: 36
Evaluacija 1/80 | cilj 17.044516787 | gradijent 7.678e-01 | proteklo 0.2 s
Evaluacija 5/80 | cilj 16.793846788 | gradijent 6.349e-02 | proteklo 1.1 s
Evaluacija 10/80 | cilj 16.788896270 | gradijent 2.437e-02 | proteklo 2.3 s
Evaluacija 15/80 | cilj 16.784720759 | gradijent 2.057e-02 | proteklo 3.4 s
Evaluacija 20/80 | cilj 16.782512816 | gradijent 1.146e-02 | proteklo 4.5 s
Evaluacija 25/80 | cilj 16.780920872 | gradijent 3.958e-02 | proteklo 5.5 s
Evaluacija 30/80 | cilj 16.778583883 | gradijent 2.636e-02 | proteklo 6.6 s
Evaluacija 35/80 | cilj 16.776854969 | gradijent 1.592e-02 | proteklo 7.6 s
Evaluacija 40/80 | cilj 16.775843555 | gradijent 8.668e-03 | proteklo 8.7 s
Evaluacija 45/80 | cilj 16.774244471 | gradijent 7.365e-03 | proteklo 9.8 s
Evaluacija 50/80 | cilj 16.772747188 | gradijent 4.627e-02 | proteklo 10.8 s
Najbolji cilj: 16.772044890
Max|projektovani gradijent|: 1.956e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.
Validacioni NLL: 16.774227408

loto7_4712_k87.csv | kontekst 0.45 | validacija 2
Uzorci: 3297 | jacina konteksta: 0.45 | parametri: 36
Evaluacija 1/80 | cilj 16.772444160 | gradijent 3.214e-02 | proteklo 0.2 s
Evaluacija 5/80 | cilj 16.770959534 | gradijent 1.450e-02 | proteklo 1.1 s
Evaluacija 10/80 | cilj 16.768925623 | gradijent 1.638e-02 | proteklo 2.2 s
Evaluacija 15/80 | cilj 16.764409192 | gradijent 5.623e-02 | proteklo 3.3 s
Evaluacija 20/80 | cilj 16.763162090 | gradijent 8.075e-03 | proteklo 4.4 s
Evaluacija 25/80 | cilj 16.762715057 | gradijent 9.784e-03 | proteklo 5.5 s
Evaluacija 30/80 | cilj 16.761996023 | gradijent 3.079e-02 | proteklo 6.7 s
Evaluacija 35/80 | cilj 16.757806954 | gradijent 2.545e-02 | proteklo 7.8 s
Evaluacija 40/80 | cilj 16.753483382 | gradijent 1.490e-02 | proteklo 8.9 s
Evaluacija 45/80 | cilj 16.750596262 | gradijent 8.318e-03 | proteklo 10.0 s
Evaluacija 50/80 | cilj 16.748414563 | gradijent 1.028e-02 | proteklo 11.1 s
Najbolji cilj: 16.747912881
Max|projektovani gradijent|: 8.543e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.
Validacioni NLL: 16.788828362

IZBOR JACINE KONTEKSTA
Jacina 0.20 | validacioni NLL 16.757619823
Jacina 0.45 | validacioni NLL 16.781527885
Izabrana jacina konteksta: 0.20
Pri razlici do 0.000010 NLL prednost ima slabiji filter.

loto7_4712_k87.csv | odvojeni test
Uzorci: 3768 | jacina konteksta: 0.20 | parametri: 36
Evaluacija 1/80 | cilj 16.747471386 | gradijent 1.671e-02 | proteklo 0.2 s
Evaluacija 5/80 | cilj 16.747194721 | gradijent 5.242e-03 | proteklo 1.2 s
Evaluacija 10/80 | cilj 16.747049375 | gradijent 2.503e-03 | proteklo 2.3 s
Evaluacija 15/80 | cilj 16.746966764 | gradijent 1.850e-03 | proteklo 3.5 s
Evaluacija 20/80 | cilj 16.746890020 | gradijent 2.047e-03 | proteklo 4.7 s
Evaluacija 25/80 | cilj 16.746814333 | gradijent 1.644e-03 | proteklo 5.8 s
Evaluacija 30/80 | cilj 16.746771003 | gradijent 3.063e-03 | proteklo 7.1 s
Evaluacija 35/80 | cilj 16.746680099 | gradijent 2.496e-03 | proteklo 8.3 s
Evaluacija 40/80 | cilj 16.746563955 | gradijent 5.492e-03 | proteklo 9.5 s
Evaluacija 45/80 | cilj 16.746444712 | gradijent 2.790e-03 | proteklo 10.7 s
Evaluacija 50/80 | cilj 16.746354167 | gradijent 5.746e-03 | proteklo 11.9 s
Najbolji cilj: 16.746323704
Max|projektovani gradijent|: 3.881e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.

Odvojeni test NLL: 16.762710222
Test ne menja izabranu jacinu konteksta.

loto7_4712_k87.csv | zavrsni trening cele istorije
Uzorci: 4711 | jacina konteksta: 0.20 | parametri: 36
Evaluacija 1/80 | cilj 16.749758771 | gradijent 1.822e-02 | proteklo 0.4 s
Evaluacija 5/80 | cilj 16.749402243 | gradijent 4.622e-03 | proteklo 2.1 s
Evaluacija 10/80 | cilj 16.749292148 | gradijent 4.902e-03 | proteklo 4.1 s
Evaluacija 15/80 | cilj 16.749228901 | gradijent 1.727e-03 | proteklo 6.2 s
Evaluacija 20/80 | cilj 16.749164751 | gradijent 3.176e-03 | proteklo 8.2 s
Evaluacija 25/80 | cilj 16.749100973 | gradijent 4.475e-03 | proteklo 10.3 s
Evaluacija 30/80 | cilj 16.748980959 | gradijent 3.064e-03 | proteklo 12.4 s
Evaluacija 35/80 | cilj 16.748806509 | gradijent 3.625e-03 | proteklo 14.5 s
Evaluacija 40/80 | cilj 16.748684519 | gradijent 3.425e-03 | proteklo 16.5 s
Evaluacija 45/80 | cilj 16.748541098 | gradijent 4.635e-03 | proteklo 18.6 s
Evaluacija 50/80 | cilj 16.748104568 | gradijent 1.174e-02 | proteklo 20.6 s
Najbolji cilj: 16.747904004
Max|projektovani gradijent|: 2.858e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.

REZULTAT: loto7_4712_k87.csv
NEXT: 6 x 14 y 27 z 30
Jacina konteksta: 0.20
Poznatih istorijskih prelaza u kontekstu: 4711
Validacioni NLL: 16.757619823
Odvojeni test NLL: 16.762710222
Modelna verovatnoca NEXT: 1.22759734451e-06
Zbir svih verovatnoca: 1.000000000000
Greska logaritamske normalizacije: 1.776e-15
Zbir verovatnoca ukljucivanja: 7.000000000000
Kombinacija obuhvacenih raspodelom: 15,380,937
Maksimum je tacan za naucenu raspodelu proizvoda Bornovih tezina uz kardinalnost sedam.
Verovatnoca pomocne grane filtera: 0.642008330
Greska unitarnosti memorijskog kola: 1.332e-15
Greska eksplicitnih Qiskit amplituda: 1.804e-16
Greska eksplicitnih Bornovih tezina: 6.245e-17
Promena raspodele uklanjanjem konteksta: 1.655292865e-02
To je dijagnostika uticaja konteksta uz iste uglove, a ne dokaz poboljsanja predikcije.
Osetljivost na istoriju i upit: 1.822879811e-01
Zavrsni projektovani gradijent: 2.858e-02
Test histogram pogodaka:
0: 206
1: 385
2: 251
3: 94
4: 7
5: 0
6: 0
7: 0
Prosecan broj pogodaka na testu: 1.269353
Verovatnoce ukljucivanja u NEXT raspodeli:
 1: 0.196945236
 2: 0.134502537
 3: 0.188301252
 4: 0.135388267
 5: 0.194059087
 6: 0.246219862
 7: 0.220191206
 8: 0.156583432
 9: 0.242788306
10: 0.148216598
11: 0.141818388
12: 0.143789690
13: 0.223516636
14: 0.244415777
15: 0.180519833
16: 0.147640625
17: 0.164518931
18: 0.165814053
19: 0.232218694
20: 0.122439504
21: 0.193378406
22: 0.210050292
23: 0.156113455
24: 0.168147690
25: 0.121825573
26: 0.151536830
27: 0.261236447
28: 0.161138753
29: 0.253805286
30: 0.297096218
31: 0.145277627
32: 0.156227205
33: 0.154174543
34: 0.166178528
35: 0.134648684
36: 0.158165732
37: 0.179456064
38: 0.147650482
39: 0.154004273
Sacuvani rezultat i kontekst: /Users/4c/Desktop/GHQ/Astra/Quantum_Tabular_InContext_Regression/Quantum_Tabular_InContext_Regression_V1_checkpoints/result_fdfb3af53d855c3ec4bae171f4a7aac3e7f30d58fd246ed11b4e780c34068ddd.npz
DIJAGNOSTIKA: NEXT ponavlja istorijsku kombinaciju.
DIJAGNOSTIKA: Stacionarnost zavrsnog treninga nije potvrdjena.
DIJAGNOSTIKA: Test model ima nepotvrdjenu stacionarnost.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
PRIPREMA KONTEKSTA: loto7_4712_k87_loto_2978.csv
Istorijskih prelaza: 2977
Poznatih prelaza u NEXT kontekstu: 2977

loto7_4712_k87_loto_2978.csv | kontekst 0.20 | validacija 1
Uzorci: 1786 | jacina konteksta: 0.20 | parametri: 36
Evaluacija 1/80 | cilj 16.994771309 | gradijent 7.156e-01 | proteklo 0.1 s
Evaluacija 5/80 | cilj 16.758413231 | gradijent 5.767e-02 | proteklo 0.6 s
Evaluacija 10/80 | cilj 16.754388116 | gradijent 1.537e-02 | proteklo 1.1 s
Evaluacija 15/80 | cilj 16.752537987 | gradijent 1.869e-02 | proteklo 1.7 s
Evaluacija 20/80 | cilj 16.750594271 | gradijent 4.546e-02 | proteklo 2.2 s
Evaluacija 25/80 | cilj 16.749128938 | gradijent 1.020e-02 | proteklo 2.8 s
Evaluacija 30/80 | cilj 16.748358922 | gradijent 9.431e-03 | proteklo 3.4 s
Evaluacija 35/80 | cilj 16.747096364 | gradijent 6.266e-03 | proteklo 3.9 s
Evaluacija 40/80 | cilj 16.746006677 | gradijent 1.883e-02 | proteklo 4.5 s
Evaluacija 45/80 | cilj 16.745156534 | gradijent 1.270e-02 | proteklo 5.1 s
Evaluacija 50/80 | cilj 16.744617341 | gradijent 3.929e-03 | proteklo 5.6 s
Najbolji cilj: 16.744451416
Max|projektovani gradijent|: 1.031e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.
Validacioni NLL: 16.745194934

loto7_4712_k87_loto_2978.csv | kontekst 0.20 | validacija 2
Uzorci: 2083 | jacina konteksta: 0.20 | parametri: 36
Evaluacija 1/80 | cilj 16.744586967 | gradijent 3.461e-02 | proteklo 0.2 s
Evaluacija 5/80 | cilj 16.743807772 | gradijent 5.889e-03 | proteklo 1.0 s
Evaluacija 10/80 | cilj 16.743517798 | gradijent 9.040e-03 | proteklo 1.9 s
Evaluacija 15/80 | cilj 16.742898624 | gradijent 8.435e-03 | proteklo 2.9 s
Evaluacija 20/80 | cilj 16.741966822 | gradijent 1.275e-02 | proteklo 3.8 s
Evaluacija 25/80 | cilj 16.740185064 | gradijent 2.260e-02 | proteklo 4.8 s
Evaluacija 30/80 | cilj 16.738009706 | gradijent 2.714e-02 | proteklo 5.7 s
Evaluacija 35/80 | cilj 16.734114505 | gradijent 5.123e-02 | proteklo 6.7 s
Evaluacija 40/80 | cilj 16.727132705 | gradijent 2.183e-02 | proteklo 7.7 s
Evaluacija 45/80 | cilj 16.724520723 | gradijent 1.339e-02 | proteklo 8.6 s
Evaluacija 50/80 | cilj 16.723542944 | gradijent 5.065e-03 | proteklo 9.6 s
Evaluacija 55/80 | cilj 16.722780889 | gradijent 8.233e-03 | proteklo 10.5 s
Najbolji cilj: 16.722554777
Max|projektovani gradijent|: 1.190e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.
Validacioni NLL: 16.731512349

loto7_4712_k87_loto_2978.csv | kontekst 0.45 | validacija 1
Uzorci: 1786 | jacina konteksta: 0.45 | parametri: 36
Evaluacija 1/80 | cilj 17.036524226 | gradijent 7.363e-01 | proteklo 0.1 s
Evaluacija 5/80 | cilj 16.791794751 | gradijent 5.916e-02 | proteklo 0.6 s
Evaluacija 10/80 | cilj 16.785855072 | gradijent 4.278e-02 | proteklo 1.1 s
Evaluacija 15/80 | cilj 16.777433292 | gradijent 5.920e-02 | proteklo 1.7 s
Evaluacija 20/80 | cilj 16.774763132 | gradijent 3.327e-02 | proteklo 2.2 s
Evaluacija 25/80 | cilj 16.773034716 | gradijent 1.153e-02 | proteklo 2.8 s
Evaluacija 30/80 | cilj 16.770863841 | gradijent 1.243e-02 | proteklo 3.4 s
Evaluacija 35/80 | cilj 16.769442663 | gradijent 1.299e-02 | proteklo 3.9 s
Evaluacija 40/80 | cilj 16.768149487 | gradijent 2.189e-02 | proteklo 4.5 s
Evaluacija 45/80 | cilj 16.763914223 | gradijent 5.970e-02 | proteklo 5.1 s
Evaluacija 50/80 | cilj 16.756464942 | gradijent 3.209e-02 | proteklo 5.6 s
Evaluacija 55/80 | cilj 16.746693719 | gradijent 2.700e-02 | proteklo 6.2 s
Najbolji cilj: 16.746693719
Max|projektovani gradijent|: 2.700e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.
Validacioni NLL: 16.758752267

loto7_4712_k87_loto_2978.csv | kontekst 0.45 | validacija 2
Uzorci: 2083 | jacina konteksta: 0.45 | parametri: 36
Evaluacija 1/80 | cilj 16.748633043 | gradijent 4.618e-02 | proteklo 0.2 s
Evaluacija 5/80 | cilj 16.744785559 | gradijent 2.392e-02 | proteklo 1.0 s
Evaluacija 10/80 | cilj 16.737856972 | gradijent 1.929e-02 | proteklo 1.9 s
Evaluacija 15/80 | cilj 16.726403436 | gradijent 5.684e-02 | proteklo 2.9 s
Evaluacija 20/80 | cilj 16.722063137 | gradijent 2.322e-02 | proteklo 3.8 s
Evaluacija 25/80 | cilj 16.719705486 | gradijent 1.768e-02 | proteklo 4.8 s
Evaluacija 30/80 | cilj 16.716886937 | gradijent 1.447e-02 | proteklo 5.8 s
Evaluacija 35/80 | cilj 16.713861085 | gradijent 1.613e-02 | proteklo 6.7 s
Evaluacija 40/80 | cilj 16.711486039 | gradijent 2.063e-02 | proteklo 7.7 s
Evaluacija 45/80 | cilj 16.709613716 | gradijent 1.468e-02 | proteklo 8.6 s
Evaluacija 50/80 | cilj 16.707840121 | gradijent 2.115e-02 | proteklo 9.6 s
Najbolji cilj: 16.705859153
Max|projektovani gradijent|: 2.186e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.
Validacioni NLL: 16.764577482

IZBOR JACINE KONTEKSTA
Jacina 0.20 | validacioni NLL 16.738342143
Jacina 0.45 | validacioni NLL 16.761669770
Izabrana jacina konteksta: 0.20
Pri razlici do 0.000010 NLL prednost ima slabiji filter.

loto7_4712_k87_loto_2978.csv | odvojeni test
Uzorci: 2381 | jacina konteksta: 0.20 | parametri: 36
Evaluacija 1/80 | cilj 16.723774227 | gradijent 1.902e-02 | proteklo 0.2 s
Evaluacija 5/80 | cilj 16.723243587 | gradijent 5.427e-03 | proteklo 1.0 s
Evaluacija 10/80 | cilj 16.722804877 | gradijent 5.506e-03 | proteklo 2.0 s
Evaluacija 15/80 | cilj 16.722000794 | gradijent 1.025e-02 | proteklo 3.0 s
Evaluacija 20/80 | cilj 16.721102621 | gradijent 8.268e-03 | proteklo 4.0 s
Evaluacija 25/80 | cilj 16.720747072 | gradijent 1.956e-02 | proteklo 5.0 s
Evaluacija 30/80 | cilj 16.720305607 | gradijent 6.193e-03 | proteklo 6.0 s
Evaluacija 35/80 | cilj 16.719866324 | gradijent 6.186e-03 | proteklo 7.0 s
Evaluacija 40/80 | cilj 16.719508564 | gradijent 4.621e-03 | proteklo 8.0 s
Evaluacija 45/80 | cilj 16.719267949 | gradijent 4.884e-03 | proteklo 9.0 s
Evaluacija 50/80 | cilj 16.718984575 | gradijent 4.621e-03 | proteklo 10.0 s
Najbolji cilj: 16.718937582
Max|projektovani gradijent|: 3.787e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.

Odvojeni test NLL: 16.728661964
Test ne menja izabranu jacinu konteksta.

loto7_4712_k87_loto_2978.csv | zavrsni trening cele istorije
Uzorci: 2977 | jacina konteksta: 0.20 | parametri: 36
Evaluacija 1/80 | cilj 16.721104610 | gradijent 2.005e-02 | proteklo 0.2 s
Evaluacija 5/80 | cilj 16.720526938 | gradijent 4.979e-03 | proteklo 1.1 s
Evaluacija 10/80 | cilj 16.720401032 | gradijent 8.610e-03 | proteklo 2.1 s
Evaluacija 15/80 | cilj 16.720294183 | gradijent 6.733e-03 | proteklo 3.2 s
Evaluacija 20/80 | cilj 16.720162911 | gradijent 3.475e-03 | proteklo 4.3 s
Evaluacija 25/80 | cilj 16.720060873 | gradijent 2.305e-03 | proteklo 5.3 s
Evaluacija 30/80 | cilj 16.719879942 | gradijent 4.492e-03 | proteklo 6.4 s
Evaluacija 35/80 | cilj 16.719629121 | gradijent 6.370e-03 | proteklo 7.4 s
Evaluacija 40/80 | cilj 16.719245702 | gradijent 1.355e-02 | proteklo 8.5 s
Evaluacija 45/80 | cilj 16.719068788 | gradijent 1.905e-03 | proteklo 9.6 s
Evaluacija 50/80 | cilj 16.718993505 | gradijent 2.694e-03 | proteklo 10.6 s
Najbolji cilj: 16.718857396
Max|projektovani gradijent|: 9.356e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.

REZULTAT: loto7_4712_k87_loto_2978.csv
NEXT: 6 x 14 y 22 z 32
Jacina konteksta: 0.20
Poznatih istorijskih prelaza u kontekstu: 2977
Validacioni NLL: 16.738342143
Odvojeni test NLL: 16.728661964
Modelna verovatnoca NEXT: 1.33567658916e-06
Zbir svih verovatnoca: 1.000000000000
Greska logaritamske normalizacije: 0.000e+00
Zbir verovatnoca ukljucivanja: 7.000000000000
Kombinacija obuhvacenih raspodelom: 15,380,937
Maksimum je tacan za naucenu raspodelu proizvoda Bornovih tezina uz kardinalnost sedam.
Verovatnoca pomocne grane filtera: 0.648529143
Greska unitarnosti memorijskog kola: 1.332e-15
Greska eksplicitnih Qiskit amplituda: 1.144e-16
Greska eksplicitnih Bornovih tezina: 3.816e-17
Promena raspodele uklanjanjem konteksta: 1.801463656e-02
To je dijagnostika uticaja konteksta uz iste uglove, a ne dokaz poboljsanja predikcije.
Osetljivost na istoriju i upit: 1.739319527e-01
Zavrsni projektovani gradijent: 9.356e-03
Test histogram pogodaka:
0: 125
1: 247
2: 167
3: 49
4: 7
5: 1
6: 0
7: 0
Prosecan broj pogodaka na testu: 1.276846
Verovatnoce ukljucivanja u NEXT raspodeli:
 1: 0.155364417
 2: 0.140424699
 3: 0.174930746
 4: 0.138079427
 5: 0.157292908
 6: 0.292308393
 7: 0.141454093
 8: 0.168162297
 9: 0.280895012
10: 0.155140405
11: 0.166374706
12: 0.136822805
13: 0.176114837
14: 0.229857119
15: 0.263518159
16: 0.138111938
17: 0.156925099
18: 0.188545360
19: 0.151052660
20: 0.164538484
21: 0.164502157
22: 0.247133546
23: 0.167511078
24: 0.120601597
25: 0.181354418
26: 0.146393382
27: 0.215471379
28: 0.153644111
29: 0.261090440
30: 0.179744457
31: 0.151029079
32: 0.221111998
33: 0.153316602
34: 0.182114043
35: 0.207743466
36: 0.160884841
37: 0.161593215
38: 0.176846381
39: 0.172000245
Sacuvani rezultat i kontekst: /Users/4c/Desktop/GHQ/Astra/Quantum_Tabular_InContext_Regression/Quantum_Tabular_InContext_Regression_V1_checkpoints/result_357c9a98594bd3fb2061157c1b9c64aef824752d6454af3ab53582d8927d853d.npz
DIJAGNOSTIKA: Stacionarnost zavrsnog treninga nije potvrdjena.
DIJAGNOSTIKA: Test model ima nepotvrdjenu stacionarnost.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
PRIPREMA KONTEKSTA: loto7_4712_k87_loto_plus_1734.csv
Istorijskih prelaza: 1733
Poznatih prelaza u NEXT kontekstu: 1733

loto7_4712_k87_loto_plus_1734.csv | kontekst 0.20 | validacija 1
Uzorci: 1039 | jacina konteksta: 0.20 | parametri: 36
Evaluacija 1/80 | cilj 16.992038522 | gradijent 8.116e-01 | proteklo 0.1 s
Evaluacija 5/80 | cilj 16.738950148 | gradijent 6.769e-02 | proteklo 0.5 s
Evaluacija 10/80 | cilj 16.733598022 | gradijent 6.066e-03 | proteklo 1.0 s
Evaluacija 15/80 | cilj 16.731084324 | gradijent 1.933e-02 | proteklo 1.4 s
Evaluacija 20/80 | cilj 16.730075175 | gradijent 1.072e-02 | proteklo 1.9 s
Evaluacija 25/80 | cilj 16.728844874 | gradijent 1.528e-02 | proteklo 2.4 s
Evaluacija 30/80 | cilj 16.727928452 | gradijent 5.631e-03 | proteklo 2.9 s
Evaluacija 35/80 | cilj 16.727080890 | gradijent 1.058e-02 | proteklo 3.4 s
Evaluacija 40/80 | cilj 16.726008373 | gradijent 2.159e-02 | proteklo 3.8 s
Evaluacija 45/80 | cilj 16.724614484 | gradijent 3.574e-02 | proteklo 4.3 s
Evaluacija 50/80 | cilj 16.721596170 | gradijent 1.758e-02 | proteklo 4.8 s
Najbolji cilj: 16.721335607
Max|projektovani gradijent|: 1.696e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.
Validacioni NLL: 16.784269787

loto7_4712_k87_loto_plus_1734.csv | kontekst 0.20 | validacija 2
Uzorci: 1213 | jacina konteksta: 0.20 | parametri: 36
Evaluacija 1/80 | cilj 16.730408952 | gradijent 6.103e-02 | proteklo 0.1 s
Evaluacija 5/80 | cilj 16.728194734 | gradijent 1.498e-02 | proteklo 0.5 s
Evaluacija 10/80 | cilj 16.725088958 | gradijent 9.303e-03 | proteklo 1.0 s
Evaluacija 15/80 | cilj 16.720427939 | gradijent 2.288e-02 | proteklo 1.5 s
Evaluacija 20/80 | cilj 16.714961311 | gradijent 1.536e-02 | proteklo 2.0 s
Evaluacija 25/80 | cilj 16.711459811 | gradijent 3.792e-02 | proteklo 2.5 s
Evaluacija 30/80 | cilj 16.706726603 | gradijent 1.922e-02 | proteklo 3.0 s
Evaluacija 35/80 | cilj 16.701187163 | gradijent 6.064e-02 | proteklo 3.5 s
Evaluacija 40/80 | cilj 16.695207436 | gradijent 1.964e-02 | proteklo 4.0 s
Evaluacija 45/80 | cilj 16.692536407 | gradijent 1.049e-02 | proteklo 4.5 s
Evaluacija 50/80 | cilj 16.690950502 | gradijent 5.851e-03 | proteklo 5.0 s
Najbolji cilj: 16.690819427
Max|projektovani gradijent|: 6.521e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.
Validacioni NLL: 16.723908552

loto7_4712_k87_loto_plus_1734.csv | kontekst 0.45 | validacija 1
Uzorci: 1039 | jacina konteksta: 0.45 | parametri: 36
Evaluacija 1/80 | cilj 17.029929869 | gradijent 8.218e-01 | proteklo 0.1 s
Evaluacija 5/80 | cilj 16.770151213 | gradijent 6.538e-02 | proteklo 0.5 s
Evaluacija 10/80 | cilj 16.763679110 | gradijent 3.683e-02 | proteklo 1.0 s
Evaluacija 15/80 | cilj 16.756998366 | gradijent 6.166e-02 | proteklo 1.5 s
Evaluacija 20/80 | cilj 16.752043293 | gradijent 3.694e-02 | proteklo 2.0 s
Evaluacija 25/80 | cilj 16.745806770 | gradijent 5.030e-02 | proteklo 2.5 s
Evaluacija 30/80 | cilj 16.737788631 | gradijent 2.531e-02 | proteklo 3.0 s
Evaluacija 35/80 | cilj 16.732273181 | gradijent 2.345e-02 | proteklo 3.4 s
Evaluacija 40/80 | cilj 16.725709689 | gradijent 2.634e-02 | proteklo 3.9 s
Evaluacija 45/80 | cilj 16.720661880 | gradijent 1.715e-02 | proteklo 4.4 s
Evaluacija 50/80 | cilj 16.719020004 | gradijent 6.737e-03 | proteklo 4.9 s
Najbolji cilj: 16.718365999
Max|projektovani gradijent|: 8.269e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.
Validacioni NLL: 16.805455288

loto7_4712_k87_loto_plus_1734.csv | kontekst 0.45 | validacija 2
Uzorci: 1213 | jacina konteksta: 0.45 | parametri: 36
Evaluacija 1/80 | cilj 16.730927605 | gradijent 6.093e-02 | proteklo 0.1 s
Evaluacija 5/80 | cilj 16.728977371 | gradijent 1.066e-02 | proteklo 0.5 s
Evaluacija 10/80 | cilj 16.727814850 | gradijent 8.940e-03 | proteklo 1.0 s
Evaluacija 15/80 | cilj 16.726453139 | gradijent 2.175e-02 | proteklo 1.5 s
Evaluacija 20/80 | cilj 16.723600401 | gradijent 1.764e-02 | proteklo 2.0 s
Evaluacija 25/80 | cilj 16.722488483 | gradijent 1.330e-02 | proteklo 2.6 s
Evaluacija 30/80 | cilj 16.720879044 | gradijent 1.491e-02 | proteklo 3.1 s
Evaluacija 35/80 | cilj 16.719255215 | gradijent 2.410e-02 | proteklo 3.6 s
Evaluacija 40/80 | cilj 16.716809222 | gradijent 2.051e-02 | proteklo 4.1 s
Evaluacija 45/80 | cilj 16.715505485 | gradijent 1.524e-02 | proteklo 4.7 s
Evaluacija 50/80 | cilj 16.713356239 | gradijent 2.148e-02 | proteklo 5.2 s
Najbolji cilj: 16.712144036
Max|projektovani gradijent|: 1.475e-02
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.
Validacioni NLL: 16.790395674

IZBOR JACINE KONTEKSTA
Jacina 0.20 | validacioni NLL 16.754176145
Jacina 0.45 | validacioni NLL 16.797947181
Izabrana jacina konteksta: 0.20
Pri razlici do 0.000010 NLL prednost ima slabiji filter.

loto7_4712_k87_loto_plus_1734.csv | odvojeni test
Uzorci: 1386 | jacina konteksta: 0.20 | parametri: 36
Evaluacija 1/80 | cilj 16.695067726 | gradijent 2.236e-02 | proteklo 0.1 s
Evaluacija 5/80 | cilj 16.694469606 | gradijent 6.227e-03 | proteklo 0.5 s
Evaluacija 10/80 | cilj 16.693848492 | gradijent 6.969e-03 | proteklo 1.1 s
Evaluacija 15/80 | cilj 16.693445893 | gradijent 5.514e-03 | proteklo 1.6 s
Evaluacija 20/80 | cilj 16.693162913 | gradijent 4.462e-03 | proteklo 2.1 s
Evaluacija 25/80 | cilj 16.692869329 | gradijent 9.789e-03 | proteklo 2.6 s
Evaluacija 30/80 | cilj 16.692094025 | gradijent 8.400e-03 | proteklo 3.1 s
Evaluacija 35/80 | cilj 16.691819878 | gradijent 5.650e-03 | proteklo 3.7 s
Evaluacija 40/80 | cilj 16.691471232 | gradijent 8.458e-03 | proteklo 4.2 s
Evaluacija 45/80 | cilj 16.690882401 | gradijent 1.001e-02 | proteklo 4.7 s
Evaluacija 50/80 | cilj 16.689560748 | gradijent 1.356e-02 | proteklo 5.2 s
Najbolji cilj: 16.689355902
Max|projektovani gradijent|: 9.592e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.

Odvojeni test NLL: 16.708049043
Test ne menja izabranu jacinu konteksta.

loto7_4712_k87_loto_plus_1734.csv | zavrsni trening cele istorije
Uzorci: 1733 | jacina konteksta: 0.20 | parametri: 36
Evaluacija 1/80 | cilj 16.693466286 | gradijent 2.934e-02 | proteklo 0.1 s
Evaluacija 5/80 | cilj 16.692008533 | gradijent 1.135e-02 | proteklo 0.6 s
Evaluacija 10/80 | cilj 16.691072793 | gradijent 7.716e-03 | proteklo 1.1 s
Evaluacija 15/80 | cilj 16.690659353 | gradijent 1.342e-02 | proteklo 1.7 s
Evaluacija 20/80 | cilj 16.689757082 | gradijent 1.684e-02 | proteklo 2.3 s
Evaluacija 25/80 | cilj 16.688962416 | gradijent 1.330e-02 | proteklo 2.9 s
Evaluacija 30/80 | cilj 16.688399663 | gradijent 8.353e-03 | proteklo 3.4 s
Evaluacija 35/80 | cilj 16.688042644 | gradijent 3.778e-03 | proteklo 4.0 s
Evaluacija 40/80 | cilj 16.687612007 | gradijent 4.771e-03 | proteklo 4.6 s
Evaluacija 45/80 | cilj 16.687428578 | gradijent 2.968e-03 | proteklo 5.2 s
Evaluacija 50/80 | cilj 16.687365673 | gradijent 4.341e-03 | proteklo 5.7 s
Najbolji cilj: 16.687348338
Max|projektovani gradijent|: 3.752e-03
Optimizator: STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
DIJAGNOSTIKA: stacionarnost nije potvrdjena; koristi se najbolji vazeci kandidat.

REZULTAT: loto7_4712_k87_loto_plus_1734.csv
NEXT: 3 x 13 y 32 z 36
Jacina konteksta: 0.20
Poznatih istorijskih prelaza u kontekstu: 1733
Validacioni NLL: 16.754176145
Odvojeni test NLL: 16.708049043
Modelna verovatnoca NEXT: 8.82833736946e-07
Zbir svih verovatnoca: 1.000000000000
Greska logaritamske normalizacije: 0.000e+00
Zbir verovatnoca ukljucivanja: 7.000000000000
Kombinacija obuhvacenih raspodelom: 15,380,937
Maksimum je tacan za naucenu raspodelu proizvoda Bornovih tezina uz kardinalnost sedam.
Verovatnoca pomocne grane filtera: 0.644020430
Greska unitarnosti memorijskog kola: 1.332e-15
Greska eksplicitnih Qiskit amplituda: 1.449e-16
Greska eksplicitnih Bornovih tezina: 4.163e-17
Promena raspodele uklanjanjem konteksta: 1.284948578e-02
To je dijagnostika uticaja konteksta uz iste uglove, a ne dokaz poboljsanja predikcije.
Osetljivost na istoriju i upit: 1.683615092e-01
Zavrsni projektovani gradijent: 3.752e-03
Test histogram pogodaka:
0: 71
1: 152
2: 84
3: 37
4: 3
5: 0
6: 0
7: 0
Prosecan broj pogodaka na testu: 1.276657
Verovatnoce ukljucivanja u NEXT raspodeli:
 1: 0.212643117
 2: 0.134130248
 3: 0.256834807
 4: 0.126617808
 5: 0.166455282
 6: 0.157765391
 7: 0.215656992
 8: 0.201246779
 9: 0.160678759
10: 0.177159549
11: 0.156772435
12: 0.149457451
13: 0.288418664
14: 0.195373634
15: 0.167791857
16: 0.168342219
17: 0.161594712
18: 0.179105876
19: 0.167828597
20: 0.156684860
21: 0.136862672
22: 0.182753934
23: 0.173738023
24: 0.120888249
25: 0.159935341
26: 0.260290171
27: 0.157485231
28: 0.169280524
29: 0.186221815
30: 0.187863761
31: 0.172238791
32: 0.237686899
33: 0.177508639
34: 0.226077795
35: 0.129963109
36: 0.228658485
37: 0.173852501
38: 0.149415247
39: 0.168719773
Sacuvani rezultat i kontekst: /Users/4c/Desktop/GHQ/Astra/Quantum_Tabular_InContext_Regression/Quantum_Tabular_InContext_Regression_V1_checkpoints/result_1990caaa9e1a338708f87bf087fb729dc3600dad175fcf278bac9c5aa2e61462.npz
DIJAGNOSTIKA: Stacionarnost zavrsnog treninga nije potvrdjena.
DIJAGNOSTIKA: Test model ima nepotvrdjenu stacionarnost.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
ZAVRSNI NEXT REZULTATI
loto7_4712_k87.csv | NEXT: 6 x 14 y 27 z 30 | kontekst: 0.20 | test NLL: 16.762710222
loto7_4712_k87_loto_2978.csv | NEXT: 6 x 14 y 22 z 32 | kontekst: 0.20 | test NLL: 16.728661964
loto7_4712_k87_loto_plus_1734.csv | NEXT: 3 x 13 y 32 z 36 | kontekst: 0.20 | test NLL: 16.708049043
"""



"""
Qiskit - kontekstualni kvantni model:
- Kontekst čine stvarni istorijski parovi „prethodna kola → naredna kombinacija”.
- Model uči kroz hronološke zadatke: raniji prelazi su kontekst, sledeći prelaz je cilj.
- Kvantno kolo mora koristiti i kontekst i novi upit za stvaranje izlaznih amplituda.
- Posle treninga, promena konteksta menja predikciju bez ponovnog optimizovanja uglova za svaki upit.
- Iz amplituda se formira raspodela legalnih kombinacija, zasebno za svaki CSV. 

Kontekst sastavljati od ranije poznatih parova „ulazna istorija → sledeća kombinacija”. 
Za prognozu kola model će smeti da koristi samo prelaze čiji je ishod već poznat. 
Naučeni uglovi ostaju fiksni pri predikciji, a prošireni kontekst menja izlazne amplitude.

Za trening kompaktan memorijski operator. 
Njegovo dejstvo i eksplicitnim Qiskit kolom sa pomoćnim kubitom, da se analitički proračun i kvantni izlaz slažu.

Ovo je regresija uslovne raspodele, bez klasifikatora. 
Kontekst sadrži stvarne istorijske prelaze; naučeni uglovi ostaju fiksni pri predikciji.

Nezavisna provera formule gradijenta dala je odstupanje oko 1.76e-10.
"""
